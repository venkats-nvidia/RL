# Score centering with i.i.d. sampler draws: vLLM side

`loss_fn.score_centering` adds the score-centering term of Marek & Ryabinin
([arXiv 2609.20807](https://arxiv.org/abs/2609.20807), Eq. 14) to `ClippedPGLossFn`. The sum over the
vocabulary is replaced by an unbiased Monte-Carlo estimate over `m` i.i.d. sampler draws per
generated token:

- At each generated position the sampler draws `v_j ~ q(.|prefix)`, with replacement, from the same
  processed distribution the token was sampled from, and returns `log q(v_j)`.
- The trainer gathers `log p_theta(v_j)` and adds the zero-valued surrogate
  `A_t * mean_j sg[w_j] * (log p(v_j) - sg[log p(v_j)])`, where `w_j = clamp(p_theta(v_j)/q(v_j), tis_min, tis_max)`.
- The loss value is unchanged. The gradient becomes `-A [w_y grad log p(y_t) - (1/m) sum_j w_j grad log p(v_j)]`.

Stock vLLM returns top-k logprobs, not i.i.d. draws. This directory holds the vLLM 0.25.1 patch that
adds them.

| file | what it does |
|---|---|
| `apply_vllm_score_centering_v0.25.1.py` | Builds patched copies of 4 vLLM 0.25.1 files. Checks the md5 of the inputs and never edits an install in place. |
| `vllm_score_centering_v0.25.1.diff` | The same change as a unified diff, for review. |
| `vllm_sc_mounts.sh` | Exports `NRL_SC_VLLM_MOUNTS`: read-only per-file binds of the patched copies over the image's vLLM venv. |
| `vllm_patch_check.py` | In-container check: version, mounts, verifier, draws ~ q, re-expansion, serving returns `m` entries. |

The patched files are `v1/sample/sampler.py`, `v1/sample/rejection_sampler.py` (speculative decoding /
MTP), `logprobs.py` and `entrypoints/openai/chat_completion/serving.py`. They are pure Python, with no `.so` files.

## Use

1. Build the patched copies once per vLLM install. `--src` is a directory holding the pristine `vllm/...`
   files, for example the image's `VllmAsyncGenerationWorker` site-packages:
   ```bash
   python3 tools/score_centering/apply_vllm_score_centering_v0.25.1.py \
       --src <dir containing vllm/> --out tools/score_centering/vllm_patched/v0.25.1
   ```
2. Mount the copies on every node that runs vLLM:
   ```bash
   source tools/score_centering/vllm_sc_mounts.sh   # set NRL_SC_VLLM_SITE if the image's venv path differs
   # append ${NRL_SC_VLLM_MOUNTS} to the container mounts
   ```
3. Turn it on in the config:
   ```yaml
   loss_fn:
     score_centering:
       enabled: true
       num_samples: 8          # m draws per generated token; must be <= vLLM max_logprobs
   token_capture:
     enabled: true             # the only ingestion path; leave token_capture.top_logprobs unset
   policy:
     generation:
       vllm_cfg:
         async_engine: true
         logprobs_mode: processed_logprobs   # recommended; raw is accepted only at T=1, top_p=1, top_k off
   ```
   Keep token-level TIS (`tis`, for example [0.2, 5]), `token_level_loss: true`, `use_cispo: false` and
   `use_fused_linear_logprobs: false`. Without the `score_centering` block the config dumps byte-identically.
4. On the first step, look for these lines in the driver log:
   - `[score_centering] ENABLED: m=8 ...`
   - once per step, `score_centering: N generated calls carry 8 i.i.d. sampler draws; 0 lack them`

`NRL_SC_IID_LOGPROBS` defaults to on in the patched copies. Setting it to `0` restores stock top-k
logprobs, and a centering run then refuses to start.

## Metrics

The loss metrics are normalized by all loss tokens. Divide by `sc_coverage` to get conditional means.

| metric | healthy |
|---|---|
| `train/sc_coverage` | about 1.0 (fraction of loss tokens that carry draws) |
| `train/sc_mean_weight` | about 1.0 (`E_q[clamp(p/q)]`) |
| `train/sc_clipped_fraction` | small |
| `train/sc_sample_lp_error` | about 0.03 nats (a large value means a misaligned gather or the wrong `q`) |

## Check the patch

Inside the container, with the mounts active, run the vLLM venv's python:

```bash
PYTHONPATH=<NeMo-RL checkout> TORCHDYNAMO_DISABLE=1 \
    <vllm venv>/bin/python tools/score_centering/vllm_patch_check.py
```

## Not supported

These raise an error instead of training silently:

- pre-packed Energon rows;
- training-side top-k/top-p filtering;
- `use_fused_linear_logprobs`;
- Automodel DTensor (Megatron only);
- score centering without token capture.
