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
"""NRL score centering (GDPVal V2 port): check the per-tree patched vLLM 0.25.1 files
INSIDE the container, through the same read-only bind mounts a launcher uses.

Run with the vLLM worker venv's python (CPU is fine):
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python \
      vllm_patch_check.py
with PYTHONPATH=<NeMo-RL checkout> (for the NeMo-RL side helpers) and TORCHDYNAMO_DISABLE=1.

Checks: (1) vllm==0.25.1 and every patched module resolves to the mounted path with the
switch on; (2) the NeMo-RL worker's verify_score_centering_vllm_patch() passes; (3) the
sampler's i.i.d. draws match q statistically, keep slot 0 = sampled token, repeat; (4) the
frontend container keeps multiplicities and the NeMo-RL extraction re-expands exactly the
sampler's draws (sampler -> logprobs.py -> worker, real vLLM code); (5) the chat serving
layer returns exactly m entries; (6) the rejection-sampler hunk is present.
"""
import importlib
import inspect
import os
import sys
from collections import Counter
from types import SimpleNamespace

import torch

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


import vllm  # noqa: E402

check(vllm.__version__ == "0.25.1", f"vllm version {vllm.__version__} == 0.25.1")
site = os.path.dirname(vllm.__file__)
for name, rel in (
    ("vllm.logprobs", "logprobs.py"),
    ("vllm.v1.sample.sampler", "v1/sample/sampler.py"),
    ("vllm.v1.sample.rejection_sampler", "v1/sample/rejection_sampler.py"),
    ("vllm.entrypoints.openai.chat_completion.serving", "entrypoints/openai/chat_completion/serving.py"),
):
    mod = importlib.import_module(name)
    check(os.path.realpath(mod.__file__) == os.path.realpath(os.path.join(site, rel)), f"{name} loads from {mod.__file__}")
    check(getattr(mod, "NRL_SC_IID_LOGPROBS", None) is True, f"{name}.NRL_SC_IID_LOGPROBS is True")
    src = open(mod.__file__).read()
    check("NRL score centering" in src, f"{name} file carries the patch marker (bind mount active)")

from nemo_rl.models.generation.vllm.utils import (  # noqa: E402
    extract_score_centering_draws,
    verify_score_centering_vllm_patch,
)

try:
    verify_score_centering_vllm_patch()
    check(True, "verify_score_centering_vllm_patch() passes")
except Exception as error:  # noqa: BLE001
    check(False, f"verify_score_centering_vllm_patch() raised {error}")

from vllm.logprobs import (  # noqa: E402
    Logprob,
    append_logprobs_for_next_position,
    create_sample_logprobs,
    nrl_sc_expand_iid_top_logprobs,
)
from vllm.v1.sample.sampler import Sampler  # noqa: E402

# (3) sampler draws: i.i.d. from q, with replacement
torch.manual_seed(0)
V, N, m = 50, 4000, 8
base = torch.randn(V) * 2.0
logprobs = torch.log_softmax(base, -1).expand(N, V).contiguous()
sampled = torch.multinomial(torch.exp(logprobs), 1).squeeze(-1).to(torch.int64)
out = Sampler.gather_iid_sample_logprobs(logprobs, m, sampled)
indices, lps, ranks = out.logprob_token_ids, out.logprobs, out.selected_token_ranks
check(tuple(indices.shape) == (N, m + 1) and indices.dtype == torch.int32, f"indices shape {tuple(indices.shape)} int32")
check(torch.equal(indices[:, 0].long(), sampled), "slot 0 is the sampled token")
check(torch.allclose(lps, logprobs.gather(-1, indices.long())), "slot logprobs = log q(slot token)")
q = torch.exp(logprobs[0])
freq = torch.bincount(indices[:, 1:].reshape(-1).long(), minlength=V).float() / (N * m)
max_dev = float((freq - q).abs().max())
check(max_dev < 0.01, f"empirical draw frequencies match q (max |freq-q| = {max_dev:.4f})")
rep = sum(1 for row in indices[:200, 1:].tolist() if len(set(row)) < len(row))
check(rep > 150, f"draws repeat (with replacement): {rep}/200 positions have a repeat")
topk = torch.topk(logprobs[0], m).indices.tolist()
check(any(sorted(row) != sorted(topk) for row in indices[:50, 1:].tolist()), "draws are not the top-k set")

# (4) frontend container + NeMo-RL re-expansion of real sampler output
container = create_sample_logprobs(False)
pos_ids = indices[:64].tolist()
pos_lps = lps[:64].tolist()
for ids_row, lps_row, rank in zip(pos_ids, pos_lps, ranks[:64].tolist()):
    append_logprobs_for_next_position(container, ids_row, lps_row, [None] * (m + 1), rank, m)
check(len(container) == 64 and all(isinstance(next(iter(p.values())), Logprob) for p in container), "container built by patched logprobs.py")
check(all(sum(lp.sc_count for lp in p.values()) == m for p in container), "multiplicities sum to m at every position")
check(all(p[ids_row[0]].rank == rank for p, ids_row, rank in zip(container, pos_ids, ranks[:64].tolist())), "sampled-token entry keeps its rank")
draws, draw_lps, repeats = extract_score_centering_draws(
    SimpleNamespace(outputs=[SimpleNamespace(token_ids=[r[0] for r in pos_ids], logprobs=container)]),
    num_samples=m,
)
check(all(Counter(d) == Counter(r[1:]) for d, r in zip(draws, pos_ids)), "worker re-expansion == sampler draws (multisets)")
check(all(abs(a - b) < 1e-6 for d, r, l, rl in zip(draws, pos_ids, draw_lps, pos_lps) for a, b in zip(sorted(l), sorted(rl[1:]))), "re-expanded draw logprobs == sampler log q")
check(repeats > 0, f"positions with repeated draws: {repeats}")

# (5) chat serving: exactly m entries (token-id placeholders)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # noqa: E402

entries = OpenAIServingChat._get_top_logprobs(
    SimpleNamespace(_get_decoded_token=OpenAIServingChat._get_decoded_token),
    container[0],
    m,
    None,
    True,
)
check(len(entries) == m, f"serving re-expansion returns exactly m={m} entries (got {len(entries)})")
check(Counter(int(e.token.removeprefix("token_id:")) for e in entries) == Counter(pos_ids[0][1:]), "serving entries == the position's draws")
stock = OpenAIServingChat._get_top_logprobs(
    SimpleNamespace(_get_decoded_token=OpenAIServingChat._get_decoded_token), container[0], 0, None, True
)
check(stock == [], "top_logprobs=0 (Gym capture pin) returns no entries")

# (6) rejection sampler hunk
import vllm.v1.sample.rejection_sampler as rs  # noqa: E402

src = inspect.getsource(rs.RejectionSampler._get_logprobs_tensors)
check("gather_iid_sample_logprobs" in src and "NRL_SC_IID_LOGPROBS" in src, "rejection sampler draws i.i.d. for accepted spec-decode tokens")

# (7) engine-side override used by the NeMo-RL worker on admitted calls: a request that keeps Gym's
# top_logprobs=0 pin but carries the private m gets SamplingParams.logprobs == m (same subclass code as
# vllm_worker_async.NeMoRLChatCompletionRequest).
from typing import Optional  # noqa: E402

from pydantic import PrivateAttr  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402


class _Req(ChatCompletionRequest):
    _nrl_sc_num_logprobs: Optional[int] = PrivateAttr(default=None)

    def to_sampling_params(self, *args, **kwargs):
        params = super().to_sampling_params(*args, **kwargs)
        if self._nrl_sc_num_logprobs:
            params.logprobs = int(self._nrl_sc_num_logprobs)
        return params


req = _Req(
    model="m",
    messages=[{"role": "user", "content": "hi"}],
    logprobs=True,
    top_logprobs=0,
    temperature=1.0,
    top_p=1.0,
)
plain = req.to_sampling_params(16, {})
check(plain.logprobs == 0, f"no override: SamplingParams.logprobs == top_logprobs pin (0), got {plain.logprobs}")
req._nrl_sc_num_logprobs = m
forced = req.to_sampling_params(16, {})
check(forced.logprobs == m and req.top_logprobs == 0, f"admitted call: SamplingParams.logprobs == m={m} (got {forced.logprobs}), request top_logprobs stays 0")
check("_nrl_sc_num_logprobs" not in req.model_dump(), "private attr is not part of the request schema/dump")

print(f"VLLM_PATCH_CHECK {'PASSED' if not failures else 'FAILED'} ({len(failures)} failures)", flush=True)
sys.exit(1 if failures else 0)
