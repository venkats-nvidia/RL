#!/bin/bash
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

# NRL score centering (GDPVal V2 port): pyxis/enroot mount spec for the per-tree PATCHED vLLM 0.25.1 files.
#
#   source tools/score_centering/vllm_sc_mounts.sh          # exports NRL_SC_VLLM_MOUNTS (comma-separated)
#   MOUNTS="${MOUNTS},${NRL_SC_VLLM_MOUNTS}"           # append to the launcher's --container-mounts list
#
# Each of the 4 patched files is bind-mounted READ-ONLY over the same file of the image's vLLM worker venv,
# so the engine core / GPU worker processes (sampler, rejection sampler) and the NeMo-RL async worker
# process (logprobs container, chat serving) all import the patched code. Per-file binds of this tree's own
# copies only: never a shared bind of a common vllm source dir (SIGBUS lesson, 2026-10-05), no .so files.
# Override the venv with NRL_SC_VLLM_SITE if the image differs (check: python -c 'import vllm; print(vllm.__file__)').
_sc_here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
NRL_SC_VLLM_SRC="${NRL_SC_VLLM_SRC:-${_sc_here}/vllm_patched/v0.25.1/vllm}"
NRL_SC_VLLM_SITE="${NRL_SC_VLLM_SITE:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/lib/python3.13/site-packages}"
_sc_mounts=()
for _sc_rel in v1/sample/sampler.py v1/sample/rejection_sampler.py logprobs.py entrypoints/openai/chat_completion/serving.py; do
  if ! grep -q "NRL score centering" "${NRL_SC_VLLM_SRC}/${_sc_rel}" 2>/dev/null; then
    echo "ERROR: ${NRL_SC_VLLM_SRC}/${_sc_rel} missing or unpatched (run apply_vllm_score_centering_v0.25.1.py --out ${_sc_here}/vllm_patched/v0.25.1)" >&2
    return 1 2>/dev/null || exit 1
  fi
  _sc_mounts+=("${NRL_SC_VLLM_SRC}/${_sc_rel}:${NRL_SC_VLLM_SITE}/vllm/${_sc_rel}:ro")
done
NRL_SC_VLLM_MOUNTS="$(IFS=,; echo "${_sc_mounts[*]}")"
export NRL_SC_VLLM_MOUNTS NRL_SC_VLLM_SITE NRL_SC_VLLM_SRC
unset _sc_here _sc_mounts _sc_rel
