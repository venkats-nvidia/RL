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
"""NRL score centering (GDPVal V2 port): build PATCHED COPIES of the vLLM 0.25.1 files.

Score centering (Marek & Ryabinin, arXiv 2609.20807), i.i.d. sampler-draw variant (ported from the V1 patch,
sections 1-2). Never edits a vLLM install in place: reads the pristine files from --src (a directory holding vllm/...), checks
their md5 against the vLLM 0.25.1 release (== the NeMo-RL image's
VllmAsyncGenerationWorker venv), applies exact-text anchors that must each occur exactly once, and writes the
patched copies under --out/vllm/... . The launcher then bind-mounts each copy READ-ONLY over the
same path inside the container's vLLM venv (see README.md and vllm_sc_mounts.sh next to this file).

Hunks (all behind NRL_SC_IID_LOGPROBS, default ON in the patched copies; =0 restores stock top-k):
  v1/sample/sampler.py           : top_logprobs slots hold m i.i.d. draws v_j ~ q (torch.multinomial
                                   WITH replacement over the processed/raw logprobs the token was
                                   sampled from) + log q(v_j); slot 0 stays the sampled token.
  v1/sample/rejection_sampler.py : same for speculative decoding (MTP/EAGLE): accepted-token
                                   positions draw from their own target distribution (new vs V1).
  logprobs.py                    : the per-position dict keeps draw multiplicities (Logprob.sc_count)
                                   instead of collapsing repeats; nrl_sc_expand_iid_top_logprobs.
  entrypoints/openai/chat_completion/serving.py : OpenAI top_logprobs = exactly m re-expanded draws.

Usage:
  python3 apply_vllm_score_centering_v0.25.1.py --src <dir with vllm/> --out <dir> [--diff <file>]
"""
import argparse
import difflib
import hashlib
import os
import sys

MARKER = "NRL score centering"
# md5 of the pristine vLLM 0.25.1 files (GitHub tag v0.25.1 == image venv copies).
EXPECTED_MD5 = {
    "vllm/v1/sample/sampler.py": "5f4a955a68c6eb9a5030e26ae82729a0",
    "vllm/logprobs.py": "b9beb6cb28661f6904b0ad83f1243c95",
    "vllm/entrypoints/openai/chat_completion/serving.py": "ad896788f2f25aff622226c66e598aa8",
    "vllm/v1/sample/rejection_sampler.py": "530929fa7e24fadae507e6f9c4aab587",
}

FLAG_BLOCK = '''import os as _os

# NRL score centering (arXiv 2609.20807, i.i.d. variant; GDPVal V2 port of venkats' V1 2026-09-19): when a request
# asks for top_logprobs=m the sampler returns m i.i.d. draws from its own token distribution (with replacement,
# duplicates kept) instead of the m most likely tokens. Default ON in this patched copy; NRL_SC_IID_LOGPROBS=0
# restores stock top-k (a score-centering run then refuses to start). Prompt logprobs are gathered as top-k.
NRL_SC_IID_LOGPROBS = _os.environ.get("NRL_SC_IID_LOGPROBS", "1").strip().lower() not in ("0", "false", "off", "no")
'''


class Patcher:
    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text

    def replace(self, old: str, new: str, count: int = 1) -> None:
        n = self.text.count(old)
        if n != count:
            raise SystemExit(f"[{self.rel}] anchor occurs {n}x (expected {count}):\n{old}")
        self.text = self.text.replace(old, new)

    def insert_after(self, anchor: str, new: str) -> None:
        self.replace(anchor, anchor + new)

    def insert_before(self, anchor: str, new: str) -> None:
        self.replace(anchor, new + anchor)


def patch_sampler(p: Patcher) -> None:
    p.insert_after("from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler\n", FLAG_BLOCK)
    p.replace(
        """        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
""",
        """        elif NRL_SC_IID_LOGPROBS and num_logprobs > 0:
            # NRL score centering: i.i.d. draws from the served distribution in the top-k slots.
            assert logprobs_mode in ("raw_logprobs", "processed_logprobs"), (
                f"NRL_SC_IID_LOGPROBS needs a *_logprobs logprobs_mode, got {logprobs_mode}"
            )
            logprobs_tensors = self.gather_iid_sample_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
""",
    )
    p.insert_before(
        "    @staticmethod\n    def _combine_outputs_with_spec_tokens(",
        '''    @staticmethod
    def gather_iid_sample_logprobs(
        logprobs: torch.Tensor,
        num_samples: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """NRL score centering: like gather_logprobs, but the `num_samples` extra slots hold i.i.d. draws
        v_1..v_m ~ q (the distribution `logprobs` describes, i.e. the one the token was sampled from), drawn
        WITH replacement, together with log q(v_j). Slot 0 is still the sampled token, its logprob and rank.

        Args:
          logprobs: (num tokens) x (vocab) float log-probabilities of the served sampling distribution
          num_samples: number m of i.i.d. draws per position
          token_ids: sampled token ids, 1D int64 (num tokens)
        Returns:
          LogprobsTensors(indices [N, m+1] int32, logprobs [N, m+1], sampled-token ranks [N])
        """
        assert token_ids.dtype == torch.int64
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        # Same dynamo hints as gather_logprobs (avoid batch-size 0/1 recompiles).
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)
        probs = torch.exp(logprobs.to(torch.float32))
        samples = torch.multinomial(probs, num_samples, replacement=True)  # [N, m]
        sample_logprobs = logprobs.gather(-1, samples)
        indices = torch.cat((token_ids, samples), dim=1).to(torch.int32)
        out_logprobs = torch.cat((token_logprobs, sample_logprobs), dim=1)
        return LogprobsTensors(indices, out_logprobs, token_ranks)

''',
    )


def patch_rejection_sampler(p: Patcher) -> None:
    p.replace(
        "from vllm.v1.sample.sampler import Sampler\n",
        "from vllm.v1.sample.sampler import (  # NRL score centering: shared i.i.d.-draw switch\n"
        "    NRL_SC_IID_LOGPROBS,\n    Sampler,\n)\n",
    )
    p.replace(
        """        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )
""",
        """        if NRL_SC_IID_LOGPROBS and max_num_logprobs > 0:
            # NRL score centering: each accepted position draws its m i.i.d. samples from the target
            # distribution that position's token was accepted under (rejection sampling makes the
            # accepted token exactly target-distributed), not the top-k.
            assert not self.is_logits_logprobs_mode, (
                "NRL_SC_IID_LOGPROBS needs a *_logprobs logprobs_mode under speculative decoding"
            )
            return self.sampler.gather_iid_sample_logprobs(
                accepted_logprobs,
                max_num_logprobs,
                accepted_tokens.to(torch.int64),
            )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )
""",
    )


def patch_logprobs(p: Patcher) -> None:
    p.insert_after("from typing import overload\n", FLAG_BLOCK)
    p.replace(
        """    decoded_token: str | None = None


LogprobsOnePosition = dict[int, Logprob]
""",
        """    decoded_token: str | None = None
    # NRL score centering: multiplicity of this token among the i.i.d. sampler draws of the position (0 = not a draw)
    sc_count: int = 0


LogprobsOnePosition = dict[int, Logprob]
""",
    )
    p.insert_before(
        "    topk_ranks = range(1, num_logprobs + 1)\n    ranks = itertools.chain((rank,), topk_ranks)\n",
        '''    if NRL_SC_IID_LOGPROBS and num_logprobs > 0:
        # NRL score centering: slot 0 = sampled token, slots 1..m = i.i.d. sampler draws (duplicates allowed). A dict
        # keyed by token id would collapse repeats, so multiplicities are kept in Logprob.sc_count and re-expanded by
        # the serving layer (nrl_sc_expand_iid_top_logprobs) / the NeMo-RL worker.
        assert not isinstance(request_logprobs, FlatLogprobs), (
            "NRL_SC_IID_LOGPROBS does not support flat_logprobs containers"
        )
        position: dict[int, Logprob] = {}
        for idx, (token_id, logprob, token) in enumerate(
            zip(token_ids, logprobs, decoded_tokens)
        ):
            if idx > num_logprobs:
                break
            if idx == 0:
                position[token_id] = Logprob(
                    logprob=logprob, rank=rank, decoded_token=token
                )
                continue
            entry = position.get(token_id)
            if entry is None:
                entry = Logprob(logprob=logprob, rank=None, decoded_token=token)
                position[token_id] = entry
            entry.sc_count += 1
        request_logprobs.append(position)
        return
''',
    )
    p.text = p.text.rstrip("\n") + '''


def nrl_sc_expand_iid_top_logprobs(logprobs_one_position, make_entry):
    """NRL score centering: re-expand the i.i.d. sampler draws stored with multiplicities (Logprob.sc_count) into a
    flat list of exactly m entries (order irrelevant, draws are i.i.d.). `make_entry(token_id, logprob_obj)` builds
    one output object; repeated draws share it."""
    out = []
    for token_id, lp in logprobs_one_position.items():
        count = getattr(lp, "sc_count", 0)
        if count <= 0:
            continue
        entry = make_entry(token_id, lp)
        out.extend([entry] * count)
    return out
'''


def patch_serving(p: Patcher) -> None:
    p.replace(
        "from vllm.logprobs import Logprob\n",
        "from vllm.logprobs import (  # NRL score centering: i.i.d. draw expansion\n"
        "    NRL_SC_IID_LOGPROBS,\n    Logprob,\n    nrl_sc_expand_iid_top_logprobs,\n)\n",
    )
    p.replace(
        """    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
""",
        """    ) -> list[ChatCompletionLogProb]:
        if NRL_SC_IID_LOGPROBS and top_logprobs and top_logprobs != -1:
            # NRL score centering: the "top" slots hold i.i.d. sampler draws with multiplicities -> exactly m entries.
            def _make_entry(token_id: int, lp: Logprob) -> ChatCompletionLogProb:
                token = self._get_decoded_token(
                    lp,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                return ChatCompletionLogProb(
                    token=token,
                    logprob=max(lp.logprob, -9999.0),
                    bytes=list(token.encode("utf-8", errors="replace")),
                )

            return nrl_sc_expand_iid_top_logprobs(logprobs, _make_entry)
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
""",
    )


PATCHES = {
    "vllm/v1/sample/sampler.py": patch_sampler,
    "vllm/v1/sample/rejection_sampler.py": patch_rejection_sampler,
    "vllm/logprobs.py": patch_logprobs,
    "vllm/entrypoints/openai/chat_completion/serving.py": patch_serving,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing pristine vllm/... files")
    ap.add_argument("--out", required=True, help="directory to write patched vllm/... copies into")
    ap.add_argument("--diff", default=None, help="optional path for the unified diff")
    args = ap.parse_args()
    diffs = []
    for rel, fn in PATCHES.items():
        src_path = os.path.join(args.src, rel)
        with open(src_path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 != EXPECTED_MD5[rel]:
            raise SystemExit(f"{src_path}: md5 {md5} != expected vLLM 0.25.1 {EXPECTED_MD5[rel]}")
        text = raw.decode("utf-8")
        if MARKER in text:
            raise SystemExit(f"{src_path} already contains '{MARKER}' (expected a pristine copy)")
        p = Patcher(rel, text)
        fn(p)
        out_path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(p.text)
        added = p.text.count("\n") - text.count("\n")
        print(f"patched {rel}  (+{added} lines) -> {out_path}")
        diffs.extend(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                p.text.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
    if args.diff:
        with open(args.diff, "w") as f:
            f.writelines(diffs)
        print(f"diff -> {args.diff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
