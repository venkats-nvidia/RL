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
"""CPU unit tests for score centering (arXiv 2609.20807, i.i.d. sampler-draw variant).

GDPVal V2 port onto NeMo-RL 5e3d27cb. Ported from the main-branch port's 26
tests (nemorl_v2_sc) and V1's CPU test, adapted to this base (loss input prep
in ``loss/utils.py``, tensor ``cu_seqlens``, ``.item()`` loss metrics, token
capture only), plus base-specific tests:

* loss: value unchanged, gradient equals the analytic centered REINFORCE+TIS
  gradient, A<0 pushes draws up / A>0 pushes them down, all-masked == baseline,
  live-PPO-ratio gating (clipped and dual-clipped tokens are not centered),
  construction-time validation and the metric registry;
* draw gather: packed and unpacked variants against a naive log-softmax
  reference (values AND gradients), the per-token packer, both prepare hooks,
  and BOTH sequence-packing wrappers (fuse_loss false/true) against the
  unpacked loss (value and gradient);
* config: the block rides ``extra="allow"``; a config without it dumps
  byte-identically; setup refuses unsupported run shapes;
* schema / data plane: conditional field registration, token-aligned jagged
  round trip of the ``[S, m]`` columns, train-field list;
* token capture: engine draw re-expansion (refusal of unpatched / top-k
  containers and count mismatches), per-call delta layout, chain assembly,
  the worker's vLLM patch verifier.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.score_centering import (
    assemble_score_centering_row,
    build_score_centering_delta,
    expand_iid_draws_from_position_logprobs,
    score_centering_config,
    score_centering_num_samples,
    sc_iid_env_enabled,
)
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    SC_FIELDS,
    SC_ROLLOUT_SCHEMA_FIELDS,
    SC_SAMPLE_LOGPROBS_FIELD,
    SC_SAMPLE_MASK_FIELD,
    SC_SAMPLE_TOKEN_IDS_FIELD,
    SC_STAGING_FIELDS,
    fields_with_optional_score_centering,
)
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_logprobs_packed_sequences,
    from_parallel_logits_to_sample_logprobs,
    from_parallel_logits_to_sample_logprobs_packed_sequences,
    get_next_token_sample_logprobs_from_logits,
)

K = 3  # draws per token in the small tests


def _no_all_reduce(monkeypatch):
    """TP=1 without a process group: the distributed log-softmax's all-reduces are no-ops."""
    import torch.distributed.nn.functional as dist_nn_functional

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dist_nn_functional, "all_reduce", lambda tensor, *args, **kwargs: tensor
    )


# --------------------------------------------------------------------------- config


def _base_loss_cfg(**overrides):
    cfg = dict(
        reference_policy_kl_penalty=0.0,
        reference_policy_kl_type="k3",
        kl_input_clamp_value=None,
        kl_output_clamp_value=None,
        ratio_clip_min=0.1,
        ratio_clip_max=0.28,
        ratio_clip_c=None,
        use_on_policy_kl_approximation=True,
        use_importance_sampling_correction=True,
        truncated_importance_sampling_ratio=5,
        truncated_importance_sampling_ratio_min=0.2,
        truncated_importance_sampling_type="tis",
        sequence_level_importance_ratios=False,
        token_level_loss=True,
        force_on_policy_ratio=True,
        use_kl_in_reward=False,
    )
    cfg.update(overrides)
    return cfg


def test_score_centering_config_parsing_dict_and_pydantic():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig

    assert score_centering_num_samples(None) == 0
    assert score_centering_num_samples({}) == 0
    assert score_centering_num_samples({"score_centering": {"enabled": False}}) == 0
    assert (
        score_centering_num_samples(
            {"score_centering": {"enabled": True, "num_samples": 8}}
        )
        == 8
    )
    with pytest.raises(ValueError):
        score_centering_num_samples({"score_centering": {"enabled": True}})
    # tkonuk's top-k variant used a bare bool; this port refuses it loudly
    with pytest.raises(TypeError):
        score_centering_num_samples({"score_centering": True})
    cfg = ClippedPGLossConfig(
        **_base_loss_cfg(score_centering={"enabled": True, "num_samples": 8})
    )
    assert score_centering_config(cfg) == {"enabled": True, "num_samples": 8}
    assert score_centering_num_samples(cfg) == 8
    assert sc_iid_env_enabled({}) is True
    assert sc_iid_env_enabled({"NRL_SC_IID_LOGPROBS": "0"}) is False


def test_config_without_block_dumps_unchanged():
    """The block is NOT a declared field: a run without it keeps its model_dump() (fingerprint)."""
    from nemo_rl.algorithms.loss import ClippedPGLossConfig

    assert "score_centering" not in ClippedPGLossConfig.model_fields
    plain = ClippedPGLossConfig(**_base_loss_cfg()).model_dump()
    assert "score_centering" not in plain
    with_block = ClippedPGLossConfig(
        **_base_loss_cfg(score_centering={"enabled": True, "num_samples": 8})
    ).model_dump()
    assert with_block["score_centering"] == {"enabled": True, "num_samples": 8}
    assert {k: v for k, v in with_block.items() if k != "score_centering"} == plain


# --------------------------------------------------------------------------- loss


def _make_loss_batch(seed=1, B=2, S=7, V=23, prev_noise=0.0):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    torch.manual_seed(seed)
    theta = torch.randn(B, S, V, requires_grad=True)  # trainer logits
    input_ids = torch.randint(0, V, (B, S))
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    sample_mask = torch.ones(B)
    adv = torch.tensor([[0.7] * S, [-1.3] * S], dtype=torch.float32)
    gen_lp = torch.log_softmax(theta.detach() + 0.8 * torch.randn(B, S, V), -1)  # mismatched q
    gen_tok_lp = gen_lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    draws = torch.randint(0, V, (B, S, K))
    draw_q = gen_lp.gather(-1, draws)
    sc_mask_msg = token_mask.clone()
    sc_mask_msg[0, 3] = 0  # one loss token without draws -> excluded from centering only
    lsm = torch.log_softmax(theta, -1)
    curr = lsm[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, S-1]
    curr_s = lsm[:, :-1].gather(-1, draws[:, 1:])  # [B, S-1, K]
    prev_tok = torch.zeros(B, S)
    prev_tok[:, 1:] = curr.detach() + prev_noise * torch.randn(B, S - 1)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([S, S]),
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": adv,
            "generation_logprobs": gen_tok_lp,
            "prev_logprobs": prev_tok,
            "reference_policy_logprobs": gen_tok_lp.clone(),
            SC_SAMPLE_TOKEN_IDS_FIELD: draws.to(torch.int32),
            SC_SAMPLE_LOGPROBS_FIELD: draw_q,
            SC_SAMPLE_MASK_FIELD: sc_mask_msg,
        }
    )
    n_tok = (token_mask[:, 1:] * sample_mask[:, None]).sum()
    return SimpleNamespace(
        theta=theta,
        data=data,
        curr=curr,
        curr_s=curr_s,
        n_tok=n_tok,
        B=B,
        adv=adv,
        token_mask=token_mask,
        sample_mask=sample_mask,
        gen_tok_lp=gen_tok_lp,
        draw_q=draw_q,
        sc_mask_msg=sc_mask_msg,
        draws=draws,
        input_ids=input_ids,
        prev_tok=prev_tok,
    )


def _losses(batch, **cfg_overrides):
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    base = ClippedPGLossFn(ClippedPGLossConfig(**_base_loss_cfg(**cfg_overrides)))
    sc = ClippedPGLossFn(
        ClippedPGLossConfig(
            **_base_loss_cfg(
                score_centering={"enabled": True, "num_samples": K}, **cfg_overrides
            )
        )
    )
    gvs = torch.tensor(float(batch.B))
    loss_b, met_b = base(
        next_token_logprobs=batch.curr,
        data=batch.data,
        global_valid_seqs=gvs,
        global_valid_toks=batch.n_tok,
    )
    loss_s, met_s = sc(
        next_token_logprobs=batch.curr,
        data=batch.data,
        global_valid_seqs=gvs,
        global_valid_toks=batch.n_tok,
        next_token_sample_logprobs=batch.curr_s,
    )
    return base, sc, loss_b, met_b, loss_s, met_s


def test_loss_value_unchanged_and_gradient_matches_analytic():
    batch = _make_loss_batch()
    _, sc_fn, loss_b, met_b, loss_s, met_s = _losses(batch)
    gb = torch.autograd.grad(loss_b, batch.theta, retain_graph=True)[0]
    gs = torch.autograd.grad(loss_s, batch.theta, retain_graph=True)[0]
    assert abs(float(loss_b) - float(loss_s)) < 1e-6
    for key in ("sc_coverage", "sc_mean_weight", "sc_clipped_fraction", "sc_sample_lp_error"):
        assert key in met_s and key not in met_b
    # analytic: -(1/N) sum mask A [ w_t log p(y_t) - (1/K) sum_j w_j log p(v_j) ], detached weights
    with torch.no_grad():
        w_t = torch.exp(batch.curr - batch.gen_tok_lp[:, 1:]).clamp(0.2, 5)
        w_s = torch.exp(batch.curr_s - batch.draw_q[:, 1:]).clamp(0.2, 5)
    mask = batch.token_mask[:, 1:] * batch.sample_mask[:, None]
    scm = mask * batch.sc_mask_msg[:, 1:]
    adv = batch.adv[:, 1:]
    obj = -(mask * adv * w_t * batch.curr).sum() / batch.n_tok + (
        scm * adv * (w_s * batch.curr_s).mean(-1)
    ).sum() / batch.n_tok
    g_ref = torch.autograd.grad(obj, batch.theta, retain_graph=True)[0]
    assert torch.allclose(gs, g_ref, atol=1e-6)
    assert not torch.allclose(gs, gb, atol=1e-6)
    obj_b = -(mask * adv * w_t * batch.curr).sum() / batch.n_tok
    assert torch.allclose(
        gb, torch.autograd.grad(obj_b, batch.theta, retain_graph=True)[0], atol=1e-6
    )
    assert abs(met_s["sc_coverage"] - float(scm.sum() / mask.sum())) < 1e-6
    from nemo_rl.algorithms.loss.interfaces import MetricNormalizer

    for key in ("sc_coverage", "sc_mean_weight", "sc_clipped_fraction", "sc_sample_lp_error"):
        assert sc_fn.metric_normalizations[key] == MetricNormalizer.TOKENS


def test_gradient_direction_negative_advantage_pushes_draws_up():
    """The centering gradient on a draw's logit has the sign of A: descent lowers
    draws under A>0 and raises them under A<0 (the sampled token moves the other way)."""
    batch = _make_loss_batch(seed=3)
    _, _, loss_b, _, loss_s, _ = _losses(batch)
    gb = torch.autograd.grad(loss_b, batch.theta, retain_graph=True)[0]
    gs = torch.autograd.grad(loss_s, batch.theta, retain_graph=True)[0]
    delta = gs - gb
    checked = 0
    for b in range(batch.B):
        for t in range(batch.theta.shape[1] - 1):
            if batch.token_mask[b, t + 1] == 0 or batch.sc_mask_msg[b, t + 1] == 0:
                continue
            for v in set(batch.draws[b, t + 1].tolist()):
                if v == int(batch.input_ids[b, t + 1]):
                    continue  # shares its logit with the sampled token's actor term
                sign = 1.0 if batch.adv[b, t + 1] > 0 else -1.0
                assert sign * float(delta[b, t, v]) > 0, (b, t, v, float(delta[b, t, v]))
                checked += 1
    assert checked > 0


def test_all_masked_draws_fall_back_to_baseline():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    batch = _make_loss_batch()
    base, sc_fn, loss_b, _, _, _ = _losses(batch)
    gb = torch.autograd.grad(loss_b, batch.theta, retain_graph=True)[0]
    data2 = BatchedDataDict(dict(batch.data))
    data2[SC_SAMPLE_MASK_FIELD] = torch.zeros_like(batch.sc_mask_msg)
    loss_0, met_0 = sc_fn(
        next_token_logprobs=batch.curr,
        data=data2,
        global_valid_seqs=torch.tensor(float(batch.B)),
        global_valid_toks=batch.n_tok,
        next_token_sample_logprobs=batch.curr_s,
    )
    assert torch.allclose(
        torch.autograd.grad(loss_0, batch.theta, retain_graph=True)[0], gb, atol=1e-6
    )
    assert met_0["sc_coverage"] == 0.0


def test_live_ppo_ratio_gate_excludes_clipped_and_dual_clipped_tokens():
    """force_on_policy_ratio=false (the GDPVal reference): tokens whose actor term is
    PPO-clipped (or dual-clipped) carry no centering gradient; the rest are centered
    with clamp(p_theta/q) of the CURRENT policy."""
    batch = _make_loss_batch(seed=5, prev_noise=0.6)  # ratios spread around 1
    cfg = dict(force_on_policy_ratio=False, ratio_clip_c=1.5)
    _, sc_fn, loss_b, _, loss_s, met_s = _losses(batch, **cfg)
    gb = torch.autograd.grad(loss_b, batch.theta, retain_graph=True)[0]
    gs = torch.autograd.grad(loss_s, batch.theta, retain_graph=True)[0]
    assert abs(float(loss_b) - float(loss_s)) < 1e-6
    with torch.no_grad():
        prev = batch.prev_tok[:, 1:]
        adv = batch.adv[:, 1:]
        r = torch.exp(batch.curr - prev)
        rc = r.clamp(1 - 0.1, 1 + 0.28)
        loss1, loss2 = -adv * r, -adv * rc
        active = loss1 >= loss2
        dual = (adv < 0) & ((-adv * 1.5) < torch.max(loss1, loss2))
        active = active & ~dual
        w_s = torch.exp(batch.curr_s - batch.draw_q[:, 1:]).clamp(0.2, 5)
    mask = batch.token_mask[:, 1:] * batch.sample_mask[:, None]
    scm = mask * batch.sc_mask_msg[:, 1:]
    assert bool((~active & scm.bool()).any()), "fixture should clip some covered tokens"
    assert bool((active & scm.bool()).any())
    centering = (scm * active.float() * adv * (w_s * batch.curr_s).mean(-1)).sum() / batch.n_tok
    g_center = torch.autograd.grad(centering, batch.theta, retain_graph=True)[0]
    assert torch.allclose(gs - gb, g_center, atol=1e-6)
    assert "sc_mean_weight" in met_s


def test_loss_requires_sample_logprobs_when_enabled():
    batch = _make_loss_batch()
    _, sc_fn, *_ = _losses(batch)
    with pytest.raises(RuntimeError, match="next_token_sample_logprobs"):
        sc_fn(
            next_token_logprobs=batch.curr,
            data=batch.data,
            global_valid_seqs=torch.tensor(float(batch.B)),
            global_valid_toks=batch.n_tok,
        )


@pytest.mark.parametrize(
    "overrides, exc",
    [
        ({"num_samples": 0}, ValueError),
        ({"sequence_level_importance_ratios": True, "token_level_loss": False}, AssertionError),
        ({"truncated_importance_sampling_type": "icepop"}, AssertionError),
        ({"use_cispo": True, "force_on_policy_ratio": False}, AssertionError),
        ({"score_centering_extra": "typo"}, AssertionError),
    ],
)
def test_init_validations(overrides, exc):
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    block = {"enabled": True, "num_samples": K}
    cfg = _base_loss_cfg()
    for key, value in overrides.items():
        if key == "num_samples":
            block["num_samples"] = value
        elif key == "score_centering_extra":
            block["bogus"] = value
        else:
            cfg[key] = value
    cfg["score_centering"] = block
    with pytest.raises(exc):
        ClippedPGLossFn(ClippedPGLossConfig(**cfg))


def test_init_rejects_fused_linear_logprobs():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    cfg = ClippedPGLossConfig(**_base_loss_cfg(score_centering={"enabled": True, "num_samples": K}))
    with pytest.raises(AssertionError):
        ClippedPGLossFn(cfg, use_fused_linear_logprobs=True)


# --------------------------------------------------------------------------- draw gather


def test_packed_sample_gather_alignment_and_gradient(monkeypatch):
    """Position t of the output holds log p(v_{t+1,k} | prefix_t), like the token logprobs."""
    _no_all_reduce(monkeypatch)
    torch.manual_seed(0)
    V = 37
    lens = [6, 9]
    cu = torch.tensor([0, 8, 18])  # padded boundaries; tensors, as on this base
    T, S = 18, 10
    logits_packed = (torch.randn(1, T, V) * 1.5).requires_grad_(True)
    input_ids = torch.randint(0, V, (2, S))
    samples = torch.randint(0, V, (2, S, K))
    packed_samples = torch.zeros(T, K, dtype=torch.int64)
    packed_ids = torch.zeros(T, dtype=torch.int64)
    for i, (length, start) in enumerate(zip(lens, cu[:-1].tolist())):
        packed_samples[start : start + length] = samples[i, :length]
        packed_ids[start : start + length] = input_ids[i, :length]
    out = from_parallel_logits_to_sample_logprobs_packed_sequences(
        logits_packed,
        packed_samples.unsqueeze(0),
        cu,
        S,
        vocab_start_index=0,
        vocab_end_index=V,
        group=None,
        inference_only=False,
        cp_group=None,
        chunk_size=4,
    )
    ref_tok = from_parallel_logits_to_logprobs_packed_sequences(
        logits_packed,
        packed_ids.unsqueeze(0),
        cu,
        S,
        vocab_start_index=0,
        vocab_end_index=V,
        group=None,
        inference_only=False,
        cp_group=None,
        target_is_pre_rolled=False,
    )
    assert tuple(out.shape) == (2, S - 1, K)
    lsm = torch.log_softmax(logits_packed[0], -1)
    for i, (length, start) in enumerate(zip(lens, cu[:-1].tolist())):
        for t in range(length - 1):
            assert torch.allclose(out[i, t], lsm[start + t][samples[i, t + 1]], atol=1e-5)
            assert torch.allclose(ref_tok[i, t], lsm[start + t][input_ids[i, t + 1]], atol=1e-5)
    w = torch.randn(2, S - 1, K)
    for i, length in enumerate(lens):
        w[i, length - 1 :] = 0
    (out * w).sum().backward()
    g_new = logits_packed.grad.clone()
    logits_packed.grad = None
    ref_total = 0
    for i, (length, start) in enumerate(zip(lens, cu[:-1].tolist())):
        for t in range(length - 1):
            ref_total = ref_total + (lsm[start + t][samples[i, t + 1]] * w[i, t]).sum()
    ref_total.backward()
    assert torch.allclose(g_new, logits_packed.grad, atol=1e-5)


def test_unpacked_sample_gather_matches_log_softmax(monkeypatch):
    _no_all_reduce(monkeypatch)
    torch.manual_seed(0)
    B, S, V = 2, 6, 19
    logits = torch.randn(B, S, V, requires_grad=True)
    samples = torch.randint(0, V, (B, S, K))
    out = from_parallel_logits_to_sample_logprobs(
        logits, samples, vocab_start_index=0, vocab_end_index=V, tp_group=None, chunk_size=2
    )
    ref = torch.log_softmax(logits[:, :-1], -1).gather(-1, samples[:, 1:])
    assert torch.allclose(out, ref, atol=1e-5)
    nonparallel = get_next_token_sample_logprobs_from_logits(samples, logits)
    assert torch.allclose(nonparallel, ref, atol=1e-5)
    (out * torch.arange(K).float()).sum().backward()
    g1 = logits.grad.clone()
    logits.grad = None
    (ref * torch.arange(K).float()).sum().backward()
    assert torch.allclose(g1, logits.grad, atol=1e-5)


def test_pack_per_token_values_layout():
    from nemo_rl.algorithms.loss.utils import _pack_input_ids, _pack_per_token_values

    torch.manual_seed(0)
    values = torch.randint(0, 50, (2, 5, K))
    ids = torch.randint(0, 50, (2, 5))
    cu_q = torch.tensor([0, 3, 8])
    cu_q_padded = torch.tensor([0, 4, 10])
    packed = _pack_per_token_values(values, cu_q, cu_q_padded)
    packed_ids = _pack_input_ids(ids, cu_q, cu_q_padded)
    assert tuple(packed.shape) == (1, 10, K)
    assert torch.equal(packed[0, 0:3], values[0, :3])
    assert torch.equal(packed[0, 3], torch.zeros(K, dtype=values.dtype))
    assert torch.equal(packed[0, 4:9], values[1, :5])
    assert torch.equal(packed[0, 9], torch.zeros(K, dtype=values.dtype))
    assert torch.equal(packed_ids[0, 0:3], ids[0, :3]) and torch.equal(packed_ids[0, 4:9], ids[1, :5])
    assert int(packed_ids[0, 3]) == 0 and int(packed_ids[0, 9]) == 0
    # the draw gather's per-sequence roll reproduces the token packer's roll_shift=-1 layout
    rolled_ids = _pack_input_ids(ids, cu_q, cu_q_padded, roll_shift=-1)
    for start, end in ((0, 4), (4, 10)):
        assert torch.equal(rolled_ids[0, start:end], packed_ids[0, start:end].roll(-1, 0))


def _stub_token_logprobs(monkeypatch):
    # The stock non-parallel token-logprob branch moves targets to CUDA; these
    # tests exercise the score-centering hook only, so stub the token path.
    monkeypatch.setattr(
        "nemo_rl.algorithms.loss.utils.get_next_token_logprobs_from_logits",
        lambda input_ids, next_token_logits, **kwargs: torch.log_softmax(
            next_token_logits[:, :-1].float(), -1
        )
        .gather(-1, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1),
    )


def test_prepare_loss_input_adds_sample_logprobs_nonparallel(monkeypatch):
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.utils import SC_LOSS_INPUT_KEY, prepare_loss_input
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    _stub_token_logprobs(monkeypatch)
    torch.manual_seed(0)
    B, S, V = 2, 5, 11
    logits = torch.randn(B, S, V)
    input_ids = torch.randint(0, V, (B, S))
    draws = torch.randint(0, V, (B, S, K))
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": torch.ones(B, S),
            "sample_mask": torch.ones(B),
            SC_SAMPLE_TOKEN_IDS_FIELD: draws.to(torch.int32),
        }
    )
    loss_fn = SimpleNamespace(
        input_type=LossInputType.LOGPROB,
        score_centering_enabled=True,
        use_fused_linear_logprobs=False,
    )
    loss_input, _ = prepare_loss_input(logits=logits, data=data, loss_fn=loss_fn)
    ref = torch.log_softmax(logits[:, :-1], -1).gather(-1, draws[:, 1:])
    assert torch.allclose(loss_input[SC_LOSS_INPUT_KEY], ref, atol=1e-5)
    data_missing = BatchedDataDict({k: v for k, v in data.items() if k != SC_SAMPLE_TOKEN_IDS_FIELD})
    with pytest.raises(RuntimeError, match="sc_sample_token_ids"):
        prepare_loss_input(logits=logits, data=data_missing, loss_fn=loss_fn)
    loss_fn_off = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering_enabled=False)
    loss_input_off, _ = prepare_loss_input(logits=logits, data=data, loss_fn=loss_fn_off)
    assert set(loss_input_off) == {"next_token_logprobs"}


def test_prepare_packed_loss_input_adds_sample_logprobs(monkeypatch):
    _no_all_reduce(monkeypatch)
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.utils import SC_LOSS_INPUT_KEY, prepare_packed_loss_input
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    torch.manual_seed(0)
    V = 13
    lens = [3, 5]
    cu_q = torch.tensor([0, 3, 8])
    cu_q_padded = torch.tensor([0, 4, 10])
    S = 5
    input_ids = torch.randint(0, V, (2, S))
    draws = torch.randint(0, V, (2, S, K))
    logits = torch.randn(1, 10, V)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor(lens),
            "token_mask": torch.ones(2, S),
            "sample_mask": torch.ones(2),
            SC_SAMPLE_TOKEN_IDS_FIELD: draws.to(torch.int32),
        }
    )
    loss_fn = SimpleNamespace(
        input_type=LossInputType.LOGPROB,
        score_centering_enabled=True,
        use_fused_linear_logprobs=False,
    )
    loss_input, _ = prepare_packed_loss_input(
        logits=logits,
        data=data,
        loss_fn=loss_fn,
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=cu_q_padded,
        vocab_parallel_rank=0,
        vocab_parallel_group=object(),
        chunk_size=None,
    )
    out = loss_input[SC_LOSS_INPUT_KEY]
    assert tuple(out.shape) == (2, S - 1, K)
    lsm = torch.log_softmax(logits[0], -1)
    for i, (length, start) in enumerate(zip(lens, cu_q_padded[:-1].tolist())):
        for t in range(length - 1):
            assert torch.allclose(out[i, t], lsm[start + t][draws[i, t + 1]], atol=1e-5)
            assert torch.allclose(
                loss_input["next_token_logprobs"][i, t], lsm[start + t][input_ids[i, t + 1]], atol=1e-5
            )


@pytest.mark.parametrize("fuse_loss", [False, True])
def test_sequence_packing_wrappers_match_unpacked_loss(monkeypatch, fuse_loss):
    """Both packing paths (policy.sequence_packing.fuse_loss false = GDPVal reference,
    and true) give the same loss value AND logits gradient as the unpacked loss."""
    _no_all_reduce(monkeypatch)
    _stub_token_logprobs(monkeypatch)
    from functools import partial

    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.loss.utils import prepare_loss_input, prepare_packed_loss_input
    from nemo_rl.algorithms.loss.wrapper import (
        SequencePackingFusionLossWrapper,
        SequencePackingLossWrapper,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    torch.manual_seed(7)
    V, S = 17, 8
    lens = [5, 8]
    cu_q = torch.tensor([0, 5, 13])
    cu_q_padded = torch.tensor([0, 6, 14])
    logits_unpacked = torch.randn(2, S, V)
    input_ids = torch.randint(0, V, (2, S))
    draws = torch.randint(0, V, (2, S, K))
    gen_lp = torch.log_softmax(logits_unpacked + 0.5 * torch.randn(2, S, V), -1)
    token_mask = torch.zeros(2, S)
    token_mask[0, 2:5] = 1
    token_mask[1, 3:8] = 1
    sc_mask = token_mask.clone()
    sc_mask[1, 4] = 0
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor(lens),
            "token_mask": token_mask,
            "sample_mask": torch.ones(2),
            "advantages": torch.tensor([[0.9] * S, [-0.4] * S]),
            "generation_logprobs": gen_lp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1),
            "prev_logprobs": torch.zeros(2, S),
            "reference_policy_logprobs": torch.zeros(2, S),
            SC_SAMPLE_TOKEN_IDS_FIELD: draws.to(torch.int32),
            SC_SAMPLE_LOGPROBS_FIELD: gen_lp.gather(-1, draws),
            SC_SAMPLE_MASK_FIELD: sc_mask,
        }
    )
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(**_base_loss_cfg(score_centering={"enabled": True, "num_samples": K}))
    )
    gvs = torch.tensor(2.0)
    gvt = token_mask[:, 1:].sum()
    # unpacked reference (non-parallel prepare path)
    theta_u = logits_unpacked.clone().requires_grad_(True)
    li, d = prepare_loss_input(logits=theta_u, data=data, loss_fn=loss_fn)
    loss_ref, met_ref = loss_fn(data=d, global_valid_seqs=gvs, global_valid_toks=gvt, **li)
    (g_ref,) = torch.autograd.grad(loss_ref, theta_u)
    # packed logits [1, T, V]
    T = int(cu_q_padded[-1])
    packed = torch.zeros(1, T, V)
    for i, length in enumerate(lens):
        start = int(cu_q_padded[i])
        packed[0, start : start + length] = logits_unpacked[i, :length]
    theta_p = packed.clone().requires_grad_(True)
    if fuse_loss:
        wrapper = SequencePackingFusionLossWrapper(
            loss_fn=loss_fn,
            prepare_fn=partial(prepare_packed_loss_input, chunk_size=None),
            cu_seqlens_q=cu_q,
            cu_seqlens_q_padded=cu_q_padded,
            vocab_parallel_rank=0,
            vocab_parallel_group=object(),
        )
    else:
        wrapper = SequencePackingLossWrapper(
            loss_fn=loss_fn,
            prepare_fn=prepare_loss_input,
            cu_seqlens_q=cu_q,
            cu_seqlens_q_padded=cu_q_padded,
        )
    loss_p, met_p = wrapper(theta_p, BatchedDataDict(dict(data)), gvs, gvt)
    (g_p,) = torch.autograd.grad(loss_p, theta_p)
    assert abs(float(loss_p) - float(loss_ref)) < 1e-5
    for i, length in enumerate(lens):
        start = int(cu_q_padded[i])
        assert torch.allclose(g_p[0, start : start + length], g_ref[i, :length], atol=1e-5)
    for key in ("sc_coverage", "sc_mean_weight", "sc_sample_lp_error"):
        assert abs(float(met_p[key]) - float(met_ref[key])) < 1e-5, key
    # and the centering term is live in this fixture
    assert float(g_ref.abs().sum()) > 0


# --------------------------------------------------------------------------- schema / data plane


def test_fields_registered_conditionally():
    assert all(field not in DP_TRAIN_FIELDS for field in SC_FIELDS)
    assert all(field not in SC_ROLLOUT_SCHEMA_FIELDS for field in SC_FIELDS)
    assert fields_with_optional_score_centering(DP_TRAIN_FIELDS, enabled=False) == list(DP_TRAIN_FIELDS)
    with_sc = fields_with_optional_score_centering(DP_TRAIN_FIELDS, enabled=True)
    assert with_sc[: len(DP_TRAIN_FIELDS)] == list(DP_TRAIN_FIELDS)
    assert with_sc[len(DP_TRAIN_FIELDS) :] == list(SC_FIELDS)
    assert set(SC_STAGING_FIELDS).isdisjoint(SC_FIELDS)
    from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS

    assert set(SC_FIELDS) <= TOKEN_ALIGNED_FIELDS


def test_train_fields_for_step_appends_sc_fields():
    pytest.importorskip("ray")  # single_controller pulls ray
    from nemo_rl.algorithms.single_controller import _train_fields_for_step

    off = _train_fields_for_step(policy_logprobs_required=True, reference_logprobs_required=False)
    on = _train_fields_for_step(
        policy_logprobs_required=True,
        reference_logprobs_required=False,
        score_centering_required=True,
    )
    assert "reference_policy_logprobs" not in off
    assert on == (*off, *SC_FIELDS)
    assert all(field not in off for field in SC_FIELDS)


def test_jagged_round_trip_of_sc_columns():
    pytest.importorskip("tensordict")
    from nemo_rl.data_plane.codec import materialize, pack_jagged_fields
    from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS

    lengths = torch.tensor([3, 5])
    ids = torch.randint(0, 9, (2, 6, K), dtype=torch.int32)
    lps = torch.randn(2, 6, K)
    mask = torch.ones(2, 6)
    td = pack_jagged_fields(
        {
            "input_ids": torch.randint(0, 9, (2, 6)),
            SC_SAMPLE_TOKEN_IDS_FIELD: ids,
            SC_SAMPLE_LOGPROBS_FIELD: lps,
            SC_SAMPLE_MASK_FIELD: mask,
        },
        lengths=lengths,
        token_aligned_fields=TOKEN_ALIGNED_FIELDS,
    )
    out = materialize(td, layout="padded", pad_value_dict={}, pad_to_seqlen=8)
    assert tuple(out[SC_SAMPLE_TOKEN_IDS_FIELD].shape) == (2, 8, K)
    assert out[SC_SAMPLE_TOKEN_IDS_FIELD].dtype == torch.int32
    assert torch.equal(out[SC_SAMPLE_TOKEN_IDS_FIELD][0, :3], ids[0, :3])
    assert torch.equal(out[SC_SAMPLE_TOKEN_IDS_FIELD][0, 3:], torch.zeros(5, K, dtype=torch.int32))
    assert torch.allclose(out[SC_SAMPLE_LOGPROBS_FIELD][1, :5], lps[1, :5])
    assert torch.equal(out[SC_SAMPLE_MASK_FIELD][0], torch.tensor([1.0, 1, 1, 0, 0, 0, 0, 0]))


def _setup_master_config(**over):
    generation = {
        "backend": "vllm",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "vllm_cfg": {"async_engine": True},
    }
    generation.update(over.pop("generation", {}))
    return SimpleNamespace(
        loss_fn=over.pop("loss_fn", {"score_centering": {"enabled": True, "num_samples": 8}}),
        token_capture=SimpleNamespace(enabled=over.pop("token_capture", True)),
        policy={"generation": generation, "megatron_cfg": over.pop("megatron_cfg", {"enabled": True})},
    )


def test_setup_checks_refuse_unsupported_run_shapes():
    pytest.importorskip("ray")
    from nemo_rl.algorithms.single_controller_utils.setup import (
        _score_centering_setup_checks,
    )

    assert _score_centering_setup_checks(_setup_master_config(loss_fn={})) == 0
    assert _score_centering_setup_checks(_setup_master_config()) == 8  # T=1/top_p=1: raw ok
    with pytest.raises(ValueError, match="token_capture"):
        _score_centering_setup_checks(_setup_master_config(token_capture=False))
    stale = _setup_master_config()
    stale.token_capture.top_logprobs = 20  # leftover top-k variant knob
    with pytest.raises(ValueError, match="top_logprobs"):
        _score_centering_setup_checks(stale)
    with pytest.raises(ValueError, match="processed_logprobs"):
        _score_centering_setup_checks(_setup_master_config(generation={"temperature": 0.7}))
    assert (
        _score_centering_setup_checks(
            _setup_master_config(
                generation={
                    "temperature": 0.7,
                    "vllm_cfg": {"async_engine": True, "logprobs_mode": "processed_logprobs"},
                }
            )
        )
        == 8
    )
    with pytest.raises(ValueError, match="use_fused_linear_logprobs"):
        _score_centering_setup_checks(
            _setup_master_config(megatron_cfg={"enabled": True, "use_fused_linear_logprobs": True})
        )
    with pytest.raises(ValueError, match="async_engine"):
        _score_centering_setup_checks(_setup_master_config(generation={"vllm_cfg": {}}))


# --------------------------------------------------------------------------- token capture helpers


def test_expand_iid_draws_from_position_logprobs():
    lp = lambda value, count: SimpleNamespace(logprob=value, sc_count=count)  # noqa: E731
    position = {7: lp(-1.0, 2), 9: lp(-2.0, 1), 2: lp(-3.5, 0)}
    ids, lps = expand_iid_draws_from_position_logprobs(position, 3)
    assert sorted(zip(ids, lps)) == [(7, -1.0), (7, -1.0), (9, -2.0)]
    with pytest.raises(RuntimeError, match="expected 3"):
        expand_iid_draws_from_position_logprobs({7: lp(-1.0, 1)}, 3)
    # unpatched engine: no sc_count attribute at all
    with pytest.raises(RuntimeError, match="sc_count"):
        expand_iid_draws_from_position_logprobs(
            {7: SimpleNamespace(logprob=-1.0), 9: SimpleNamespace(logprob=-2.0)}, 2
        )
    # patched engine with NRL_SC_IID_LOGPROBS=0: stock top-k entries, sc_count 0 everywhere
    with pytest.raises(RuntimeError, match="expected 2"):
        expand_iid_draws_from_position_logprobs({7: lp(-1.0, 0), 9: lp(-2.0, 0), 4: lp(-2.5, 0)}, 2)
    with pytest.raises(RuntimeError):
        expand_iid_draws_from_position_logprobs(None, 2)


def test_extract_score_centering_draws_from_engine_output():
    pytest.importorskip("ray")
    from nemo_rl.models.generation.vllm.utils import extract_score_centering_draws

    lp = lambda value, count, rank=None: SimpleNamespace(logprob=value, sc_count=count, rank=rank)  # noqa: E731
    output = SimpleNamespace(
        token_ids=[5, 6],
        logprobs=[
            {5: lp(-0.1, 2, 1), 8: lp(-3.0, 1)},  # sampled token drawn twice
            {6: lp(-0.5, 0, 2), 1: lp(-0.2, 3)},
        ],
    )
    draws, draw_lps, repeats = extract_score_centering_draws(
        SimpleNamespace(outputs=[output]), num_samples=3
    )
    assert [sorted(d) for d in draws] == [[5, 5, 8], [1, 1, 1]]
    assert repeats == 2
    assert all(len(row) == 3 for row in draw_lps)
    with pytest.raises(RuntimeError, match="logprob positions"):
        extract_score_centering_draws(
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 6], logprobs=output.logprobs[:1])]),
            num_samples=3,
        )
    with pytest.raises(RuntimeError, match="exactly one output"):
        extract_score_centering_draws(SimpleNamespace(outputs=[output, output]), num_samples=3)
    assert extract_score_centering_draws(
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[], logprobs=[])]), num_samples=3
    ) == ([], [], 0)


def _fake_vllm_modules(monkeypatch, *, patched=True, flag=True):
    from dataclasses import dataclass

    @dataclass
    class Logprob:
        logprob: float
        rank: int | None = None
        decoded_token: str | None = None

    if patched:
        Logprob = dataclass(type("Logprob", (Logprob,), {"__annotations__": {"sc_count": int}, "sc_count": 0}))

    class Sampler:
        pass

    if patched:
        Sampler.gather_iid_sample_logprobs = staticmethod(lambda *a, **k: None)
    names = {
        "vllm": types.ModuleType("vllm"),
        "vllm.logprobs": types.ModuleType("vllm.logprobs"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.sample": types.ModuleType("vllm.v1.sample"),
        "vllm.v1.sample.sampler": types.ModuleType("vllm.v1.sample.sampler"),
        "vllm.v1.sample.rejection_sampler": types.ModuleType("vllm.v1.sample.rejection_sampler"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.chat_completion": types.ModuleType("vllm.entrypoints.openai.chat_completion"),
        "vllm.entrypoints.openai.chat_completion.serving": types.ModuleType(
            "vllm.entrypoints.openai.chat_completion.serving"
        ),
    }
    names["vllm.logprobs"].Logprob = Logprob
    names["vllm.v1.sample.sampler"].Sampler = Sampler
    for name in (
        "vllm.logprobs",
        "vllm.v1.sample.sampler",
        "vllm.v1.sample.rejection_sampler",
        "vllm.entrypoints.openai.chat_completion.serving",
    ):
        if patched:
            names[name].NRL_SC_IID_LOGPROBS = flag
    for name, module in names.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "patched, flag, ok",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_verify_score_centering_vllm_patch(monkeypatch, patched, flag, ok):
    pytest.importorskip("ray")
    from nemo_rl.models.generation.vllm.utils import verify_score_centering_vllm_patch

    _fake_vllm_modules(monkeypatch, patched=patched, flag=flag)
    if ok:
        verify_score_centering_vllm_patch()
    else:
        with pytest.raises(RuntimeError, match="cannot produce i.i.d. sampler draws"):
            verify_score_centering_vllm_patch()


def test_build_delta_and_assemble_row():
    ids, lps = build_score_centering_delta(
        draws=[[1, 2, 3], [4, 5, 6]], draw_logprobs=[[-1, -2, -3], [-4, -5, -6]], carry_len=2, num_samples=3
    )
    assert tuple(ids.shape) == (4, 3) and ids.dtype == torch.int32 and lps.dtype == torch.float32
    assert torch.equal(ids[:2], torch.zeros(2, 3, dtype=torch.int32))
    assert ids[2].tolist() == [1, 2, 3] and lps[3].tolist() == [-4.0, -5.0, -6.0]
    with pytest.raises(ValueError):
        build_score_centering_delta(draws=[[1, 2]], draw_logprobs=[[-1, -2]], carry_len=0, num_samples=3)
    with pytest.raises(ValueError):
        build_score_centering_delta(draws=[], draw_logprobs=[], carry_len=-1, num_samples=3)
    # chain: root call (prompt 2 + gen 2, draws), child call (carry 1 + gen 1, NO draws)
    link_spans = [("root", 2, 2), ("child", 1, 1)]
    token_mask = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    out_ids, out_lps, mask, with_draws, without = assemble_score_centering_row(
        link_spans=link_spans, token_mask=token_mask, deltas_by_call={"root": (ids, lps)}, num_samples=3
    )
    assert (with_draws, without) == (1, 1)
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert torch.equal(out_ids[:4], ids) and torch.equal(out_ids[4:], torch.zeros(2, 3, dtype=torch.int32))
    with pytest.raises(ValueError, match="link spans cover"):
        assemble_score_centering_row(link_spans=link_spans, token_mask=token_mask[:-1], deltas_by_call={}, num_samples=3)
    with pytest.raises(ValueError, match="staged draws"):
        assemble_score_centering_row(
            link_spans=link_spans, token_mask=token_mask, deltas_by_call={"root": (ids[:3], lps[:3])}, num_samples=3
        )
