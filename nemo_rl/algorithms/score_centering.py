# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NRL score centering (GDPVal V2 port) — shared helpers.

Score centering (Marek & Ryabinin, arXiv 2609.20807), i.i.d. sampler-draw
variant. Per generated token ``t`` the sampler (patched vLLM) returns ``m``
i.i.d. draws ``v_j ~ q(.|prefix)`` from the SAME processed distribution the
token was sampled from, together with ``log q(v_j)``; the trainer gathers
``log p_theta(v_j)`` at every position and ``ClippedPGLossFn`` adds a
zero-valued surrogate whose gradient subtracts the Monte-Carlo estimate of the
expected weighted score ``sum_v q_v w_v grad log p_theta(v)`` from the actor
gradient (paper Eq. 14 with the top-k sum replaced by the ``m``-draw estimate).

Data path on this base (NeMo-RL 5e3d27cb, SingleController + token capture):

* vLLM sampler (patched; ``NRL_SC_IID_LOGPROBS=1``) returns the draws in the
  ``top_logprobs`` slots; the frontend keeps repeat multiplicities in
  ``Logprob.sc_count``.
* The NeMo-RL vLLM worker forces ``top_logprobs=m`` on every admitted
  (``ng_capture``) call, re-expands the draws from the engine's per-position
  containers and hands them to ``TQTokenSink`` beside the call's token delta
  (staging columns ``sc_sample_*_delta`` + ``sc_present``).
* ``RolloutReassembler`` concatenates the per-call draws along the verified
  terminal chain into the canonical row columns ``sc_sample_token_ids``
  ``[S, m]`` int32, ``sc_sample_logprobs`` ``[S, m]`` float32 and
  ``sc_sample_mask`` ``[S]`` float32, which the trainer fetches with the other
  train fields.

The chat-echo ingestion path (``token_capture.enabled=false``) is NOT ported on
this base; setup refuses that combination.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

# Per-row tag keys (ride ``tags`` like ``payload.VIOLATION_TAG_KEYS``): how
# many terminal-chain calls of the row carried the ``m`` draws, and how many
# lacked them (their positions are masked from the centering term). Summed per
# step by the controller for the plumbing print and the no-draws guard.
SC_TAG_WITH_DRAWS = "num_sc_messages_with_draws"
SC_TAG_WITHOUT_DRAWS = "num_sc_messages_without_draws"
SC_TAG_KEYS = (SC_TAG_WITH_DRAWS, SC_TAG_WITHOUT_DRAWS)

# Error raised when every generated call of a step lacks draws although
# centering is enabled (V1 follow-up guard): the plumbing is broken, refuse to
# train the baseline under the score-centering name.
SC_NO_DRAWS_MESSAGE = (
    "score centering is enabled but NO generated call carries sampler draws "
    "({missing} lack them): the vLLM patch or the worker draw capture is not "
    "active on this run (check that the patched vLLM files are mounted, "
    "NRL_SC_IID_LOGPROBS=1, and token_capture.enabled). Refusing to train the "
    "baseline under the score-centering name."
)

# Environment switch read by the patched vLLM files (sampler, rejection
# sampler, logprobs, chat serving). The patched copies shipped with this port
# default to ON; the NeMo-RL worker additionally refuses to start a centering
# run when the switch is explicitly off.
SC_IID_ENV = "NRL_SC_IID_LOGPROBS"


def score_centering_config(loss_cfg: Any) -> dict[str, Any]:
    """Return the ``loss_fn.score_centering`` block as a plain dict (``{}`` when absent).

    Works for the pydantic ``ClippedPGLossConfig`` (the key rides
    ``extra="allow"``), for a plain ``loss_fn`` mapping, and for ``None``.
    """
    if loss_cfg is None:
        return {}
    if isinstance(loss_cfg, Mapping):
        block = loss_cfg.get("score_centering")
    else:
        block = getattr(loss_cfg, "score_centering", None)
        if block is None:
            extra = getattr(loss_cfg, "__pydantic_extra__", None)
            if isinstance(extra, Mapping):
                block = extra.get("score_centering")
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise TypeError(
            "loss_fn.score_centering must be a mapping {enabled: bool, "
            f"num_samples: int}}, got {type(block).__name__}"
        )
    return dict(block)


def score_centering_num_samples(loss_cfg: Any) -> int:
    """Number ``m`` of per-token sampler draws the loss expects, or 0 when disabled."""
    block = score_centering_config(loss_cfg)
    if not bool(block.get("enabled", False)):
        return 0
    num_samples = int(block.get("num_samples", 0) or 0)
    if num_samples <= 0:
        raise ValueError(
            "loss_fn.score_centering.num_samples must be a positive integer when "
            f"score centering is enabled, got {block.get('num_samples')!r}"
        )
    return num_samples


def sc_iid_env_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Mirror of the patched vLLM files' ``NRL_SC_IID_LOGPROBS`` parse (default ON)."""
    import os

    env = os.environ if environ is None else environ
    return env.get(SC_IID_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def expand_iid_draws_from_position_logprobs(
    position_logprobs: Any,
    num_samples: int,
) -> tuple[list[int], list[float]]:
    """Re-expand one position's i.i.d. draws from the engine's logprob dict.

    The patched vLLM sampler stores the ``m`` draws of a position in the
    ``top_logprobs`` slots; the engine's per-position container is a dict keyed
    by token id, so repeats are kept as multiplicities in ``Logprob.sc_count``
    (0 on the sampled-token entry unless it was also drawn). This is the
    in-process equivalent of vLLM's ``nrl_sc_expand_iid_top_logprobs``.

    Raises:
        RuntimeError: when the container carries no ``sc_count`` information
            (unpatched vLLM, or ``NRL_SC_IID_LOGPROBS=0``: stock top-k entries
            have ``sc_count == 0`` everywhere) or the multiplicities do not sum
            to ``num_samples`` — never silently fall back to top-k entries.
    """
    if position_logprobs is None:
        raise RuntimeError(
            "score centering: the engine returned no logprobs for a generated "
            "position (is logprobs=True/top_logprobs=m on the request?)"
        )
    draws: list[int] = []
    draw_logprobs: list[float] = []
    saw_sc_count = False
    for token_id, entry in position_logprobs.items():
        count = getattr(entry, "sc_count", None)
        if count is None:
            continue
        saw_sc_count = True
        count = int(count)
        if count <= 0:
            continue
        draws.extend([int(token_id)] * count)
        draw_logprobs.extend([float(entry.logprob)] * count)
    if not saw_sc_count:
        raise RuntimeError(
            "score centering: the engine's logprob entries carry no `sc_count` "
            "multiplicities. The patched vLLM (patches/vllm_score_centering_"
            "v0.25.1.diff, mounted per tree) is required; stock top-k logprobs "
            "are NOT i.i.d. draws and are refused."
        )
    if len(draws) != num_samples:
        raise RuntimeError(
            f"score centering: expected {num_samples} i.i.d. sampler draws per "
            f"generated token, got {len(draws)} (with NRL_SC_IID_LOGPROBS=0 the "
            "patched engine returns stock top-k entries, which carry no draws; "
            "the request top_logprobs must equal loss_fn.score_centering.num_samples)"
        )
    return draws, draw_logprobs


def build_score_centering_delta(
    *,
    draws: Sequence[Sequence[int]],
    draw_logprobs: Sequence[Sequence[float]],
    carry_len: int,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay the draws of one captured call out over its staged token delta.

    A staged delta is ``carry_len`` prompt-carry tokens (new prompt tokens
    beyond the parent's prefix, ``prompt_len - prev_len``) followed by the
    generated tokens (Gym ``build_staging_delta``); the draws exist for the
    generated tokens only, carry positions are zero-filled.

    Returns:
        ``([delta_len, m] int32, [delta_len, m] float32)``.
    """
    if carry_len < 0:
        raise ValueError(f"score centering: negative carry length {carry_len}")
    if len(draws) != len(draw_logprobs):
        raise ValueError(
            f"score centering: {len(draws)} draw rows vs {len(draw_logprobs)} logprob rows"
        )
    generated = len(draws)
    ids = torch.zeros((carry_len + generated, num_samples), dtype=torch.int32)
    lps = torch.zeros((carry_len + generated, num_samples), dtype=torch.float32)
    if generated:
        ids_t = torch.tensor([list(row) for row in draws], dtype=torch.int32)
        lps_t = torch.tensor([list(row) for row in draw_logprobs], dtype=torch.float32)
        if tuple(ids_t.shape) != (generated, num_samples) or lps_t.shape != ids_t.shape:
            raise ValueError(
                f"score centering: draws must be [{generated}, {num_samples}], got "
                f"{tuple(ids_t.shape)} / {tuple(lps_t.shape)}"
            )
        ids[carry_len:] = ids_t
        lps[carry_len:] = lps_t
    return ids, lps


def assemble_score_centering_row(
    *,
    link_spans: Sequence[tuple[str, int, int]],
    token_mask: Sequence[float],
    deltas_by_call: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Concatenate per-call draw deltas along a rebuilt terminal chain.

    Args:
        link_spans: ``(model_call_id, carry_len, generation_len)`` per selected
            call, root to terminal (Gym ``LinearizedRow.link_spans``); the row's
            tokens are the concatenation of these spans.
        token_mask: the rebuilt row's token mask (1 on generated tokens).
        deltas_by_call: staged ``([delta_len, m], [delta_len, m])`` for the
            chain calls that carried draws.
        num_samples: ``m``.

    Returns:
        ``(sc_sample_token_ids [L, m] int32, sc_sample_logprobs [L, m] float32,
        sc_sample_mask [L] float32, calls_with_draws, calls_without_draws)``
        where ``L = len(token_mask)``. Positions of calls without draws are
        zero-filled with mask 0; positions of calls with draws get the token
        mask (1 exactly on generated tokens).
    """
    length = len(token_mask)
    ids = torch.zeros((length, num_samples), dtype=torch.int32)
    lps = torch.zeros((length, num_samples), dtype=torch.float32)
    mask = torch.zeros((length,), dtype=torch.float32)
    token_mask_t = torch.tensor(list(token_mask), dtype=torch.float32)
    offset = 0
    with_draws = without_draws = 0
    for call_id, carry_len, generation_len in link_spans:
        span_len = int(carry_len) + int(generation_len)
        delta = deltas_by_call.get(call_id)
        if delta is None:
            if int(generation_len) > 0:
                without_draws += 1
        else:
            d_ids, d_lps = delta
            if tuple(d_ids.shape) != (span_len, num_samples) or d_lps.shape != d_ids.shape:
                raise ValueError(
                    f"score centering: staged draws for call {call_id} are "
                    f"{tuple(d_ids.shape)} / {tuple(d_lps.shape)}, expected "
                    f"({span_len}, {num_samples})"
                )
            ids[offset : offset + span_len] = d_ids.to(torch.int32)
            lps[offset : offset + span_len] = d_lps.to(torch.float32)
            mask[offset : offset + span_len] = token_mask_t[offset : offset + span_len]
            with_draws += 1
        offset += span_len
    if offset != length:
        raise ValueError(
            f"score centering: link spans cover {offset} tokens but the row has {length}"
        )
    return ids, lps, mask, with_draws, without_draws


def format_score_centering_plumbing_line(
    with_draws: int, without_draws: int, num_samples: int
) -> str:
    """The per-step plumbing line (V1 grpo.py wording; counts are chain calls here)."""
    return (
        f"  📊 score_centering: {with_draws} generated calls carry "
        f"{num_samples} i.i.d. sampler draws; {without_draws} lack them "
        "(masked from the centering term)"
    )
