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
"""NRL score centering (GDPVal V2 port): token-capture draw path end to end on CPU.

Worker side channel -> ``TQTokenSink.stage`` (draw columns in the same put as the
token delta) -> ``RolloutReassembler.finalize_rollout/finalize_group`` (draws
concatenated along Gym's verified terminal chain) -> canonical row columns +
per-row tags, driven by the golden Gym capture fixtures. The data plane is an
in-memory fake with TransferQueue's batching semantics (stacked when row shapes
agree, ``torch.jagged`` nested otherwise); a live-TQ variant runs when the
``NRL_SC_LIVE_TQ`` environment variable is set and TransferQueue can start.
"""

from __future__ import annotations

import os

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_rl.algorithms.score_centering import (  # noqa: E402
    SC_TAG_WITH_DRAWS,
    SC_TAG_WITHOUT_DRAWS,
)
from nemo_rl.data_plane.schema import (  # noqa: E402
    SC_FIELDS,
    SC_SAMPLE_LOGPROBS_FIELD,
    SC_SAMPLE_MASK_FIELD,
    SC_SAMPLE_TOKEN_IDS_FIELD,
    SC_STAGING_FIELDS,
    SC_STAGING_PRESENT_FIELD,
)
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    STAGING_FIELDS,
    TQTokenSink,
)
from nemo_rl.experience.rollout_reassembler import RolloutReassembler  # noqa: E402
from tests.unit.data_plane.token_capture_test_fixtures import (  # noqa: E402
    build_fixture_artifacts,
)

M = 4
STAGING = "sc_staging"
CANONICAL = "sc_canonical"


class _Rows:
    def __init__(self, data: dict, n: int):
        self._data = data
        self.batch_size = torch.Size([n])

    def keys(self):
        return self._data.keys()

    def get(self, key):
        return self._data[key]

    def __getitem__(self, key):
        return self._data[key]


class FakeDataPlane:
    """In-memory put/get/clear with TQ's row semantics (see module docstring)."""

    def __init__(self):
        self.store: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self.tags: dict[tuple[str, str], dict] = {}

    def put_samples(self, *, sample_ids, partition_id, fields, tags=None):
        for index, sample_id in enumerate(sample_ids):
            row = self.store.setdefault((partition_id, sample_id), {})
            for key in fields.keys():
                value = fields.get(key)
                row[key] = value[index].clone()
            if tags is not None:
                self.tags[(partition_id, sample_id)] = dict(tags[index])

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        rows = []
        for sample_id in sample_ids:
            if (partition_id, sample_id) not in self.store:
                continue
            rows.append(self.store[(partition_id, sample_id)])
        if not rows:
            raise KeyError(f"no rows for {sample_ids}")
        out = {}
        for field in select_fields:
            values = [row[field] for row in rows]  # KeyError on a missing column, like TQ
            if all(v.shape == values[0].shape for v in values):
                out[field] = torch.stack(values)
            else:
                out[field] = torch.nested.nested_tensor(values, layout=torch.jagged)
        return _Rows(out, len(rows))

    def clear_samples(self, *, sample_ids, partition_id):
        for sample_id in list(sample_ids or []):
            self.store.pop((partition_id, sample_id), None)


def _draws_for(record, salt: int):
    """Deterministic per-call draws laid over the delta (carry zero-filled)."""
    delta_len = len(record.token_ids_delta)
    gen = int(sum(record.token_mask_delta))
    carry = delta_len - gen
    ids = torch.zeros(delta_len, M, dtype=torch.int32)
    lps = torch.zeros(delta_len, M, dtype=torch.float32)
    for pos in range(carry, delta_len):
        ids[pos] = torch.tensor([salt * 100 + pos * M + j for j in range(M)])
        lps[pos] = -0.01 * (ids[pos].float() % 7 + 1)
    return ids, lps


def _stage(dp, rollout_id, *, m, with_draws_for=("c1", "c2"), name="worked_example"):
    records, receipt, row = build_fixture_artifacts(name, rollout_id=rollout_id)
    sink = TQTokenSink(dp, staging_partition=STAGING, score_centering_num_samples=m)
    expected = {}
    for salt, record in enumerate(records, start=1):
        if m > 0 and record.model_call_id in with_draws_for:
            ids, lps = _draws_for(record, salt)
            expected[record.model_call_id] = (ids, lps)
            sink.set_score_centering_draws(record.rollout_id, record.model_call_id, ids, lps)
        assert sink.stage(record).ok
    return records, receipt.model_dump(), row, expected


def _finalizer(dp, m):
    return RolloutReassembler(
        dp,
        partition_id=CANONICAL,
        staging_partition=STAGING,
        pad_token_id=0,
        max_seq_len=4096,
        score_centering_num_samples=m,
    )


def _expected_row(row, expected):
    ids, lps, mask = [], [], []
    for call_id, carry, gen in row.link_spans:
        span = carry + gen
        if call_id in expected:
            ids.append(expected[call_id][0])
            lps.append(expected[call_id][1])
            mask.append(torch.tensor([0.0] * carry + [1.0] * gen))
        else:
            ids.append(torch.zeros(span, M, dtype=torch.int32))
            lps.append(torch.zeros(span, M))
            mask.append(torch.zeros(span))
    return torch.cat(ids), torch.cat(lps), torch.cat(mask)


def _run_group(dp, *, with_draws_for):
    m = M
    records, receipt, row, expected = _stage(dp, "grp_g0", m=m, with_draws_for=with_draws_for)
    staged = dp.store[(STAGING, records[0].staging_key)]
    assert set(SC_STAGING_FIELDS) <= set(staged)
    finalizer = _finalizer(dp, m)
    finalized = finalizer.finalize_group(
        "grp",
        ["grp_g0", "grp_g1"],
        [receipt, None],  # second rollout lost its receipt -> placeholder row
        [1.0, 0.0],
        mask_sample=[False, False],
        fallback_weight_version=9,
        prompt_idx=3,
    )
    assert not finalized.dropped
    rows = dp.get_samples(
        sample_ids=["grp_g0", "grp_g1"], partition_id=CANONICAL, select_fields=list(SC_FIELDS)
    )
    exp_ids, exp_lps, exp_mask = _expected_row(row, expected)
    got_ids = rows.get(SC_SAMPLE_TOKEN_IDS_FIELD)[0]
    got_lps = rows.get(SC_SAMPLE_LOGPROBS_FIELD)[0]
    got_mask = rows.get(SC_SAMPLE_MASK_FIELD)[0]
    assert torch.equal(got_ids, exp_ids)
    assert torch.allclose(got_lps, exp_lps)
    assert torch.equal(got_mask, exp_mask)
    # placeholder row: all-zero columns
    assert float(rows.get(SC_SAMPLE_MASK_FIELD)[1].abs().sum()) == 0.0
    return finalized, row


def test_capture_draws_reach_the_canonical_row():
    dp = FakeDataPlane()
    finalized, row = _run_group(dp, with_draws_for=("c1", "c2"))
    tags = finalized.meta.tags
    assert tags[0][SC_TAG_WITH_DRAWS] == len(row.link_spans)
    assert tags[0][SC_TAG_WITHOUT_DRAWS] == 0
    assert tags[1][SC_TAG_WITH_DRAWS] == 0 and tags[1][SC_TAG_WITHOUT_DRAWS] == 0
    assert finalized.metrics["finalize/score_centering_calls_with_draws_rate"] == 1.0
    assert set(SC_FIELDS) <= set(finalized.meta.fields)


def test_capture_call_without_draws_is_masked_not_rejected():
    dp = FakeDataPlane()
    finalized, row = _run_group(dp, with_draws_for=("c1",))
    tags = finalized.meta.tags
    assert (tags[0][SC_TAG_WITH_DRAWS], tags[0][SC_TAG_WITHOUT_DRAWS]) == (1, 1)
    assert finalized.metrics["finalize/score_centering_calls_with_draws_rate"] == 0.5
    assert finalized.valid_row_count == 1


def test_capture_disabled_path_is_unchanged():
    dp = FakeDataPlane()
    records, receipt, row, _ = _stage(dp, "off_g0", m=0)
    staged = dp.store[(STAGING, records[0].staging_key)]
    assert set(staged) == set(STAGING_FIELDS)  # no score-centering columns
    finalizer = _finalizer(dp, 0)
    finalized = finalizer.finalize_group(
        "off",
        ["off_g0"],
        [receipt],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=0,
        prompt_idx=0,
    )
    assert not any(field in finalized.meta.fields for field in SC_FIELDS)
    assert SC_TAG_WITH_DRAWS not in finalized.meta.tags[0]
    assert not any(k.startswith("finalize/score_centering") for k in finalized.metrics)
    sink = TQTokenSink(dp, staging_partition=STAGING)
    with pytest.raises(ValueError):
        sink.set_score_centering_draws("r", "c", torch.zeros(1, M), torch.zeros(1, M))


def test_stage_poisons_a_call_whose_draws_do_not_cover_the_delta():
    dp = FakeDataPlane()
    records, _, _ = build_fixture_artifacts("single_call", rollout_id="bad_r0")
    sink = TQTokenSink(dp, staging_partition=STAGING, score_centering_num_samples=M)
    record = records[0]
    sink.set_score_centering_draws(
        record.rollout_id, record.model_call_id, torch.zeros(1, M, dtype=torch.int32), torch.zeros(1, M)
    )
    result = sink.stage(record)
    assert not result.ok and "draws cover" in (result.error or "")
    # the side-channel entry was consumed, a retry stages sentinels
    assert sink.stage(record).ok
    staged = dp.store[(STAGING, record.staging_key)]
    assert bool(staged[SC_STAGING_PRESENT_FIELD]) is False


def test_side_channel_discard():
    dp = FakeDataPlane()
    sink = TQTokenSink(dp, staging_partition=STAGING, score_centering_num_samples=M)
    key = sink.set_score_centering_draws("r", "c", torch.zeros(2, M), torch.zeros(2, M))
    sink.discard_score_centering_draws(key)
    assert key not in sink._sc_pending


@pytest.mark.skipif(
    not os.environ.get("NRL_SC_LIVE_TQ"),
    reason="set NRL_SC_LIVE_TQ=1 (and load tests/sc_tq_plugin.py) to run on a live TQ simple backend",
)
def test_capture_draws_on_live_transfer_queue(request):
    client = request.getfixturevalue("tq_client")  # live TQ simple backend (sc_tq_plugin)
    if True:
        client.register_partition(
            partition_id=STAGING,
            fields=list(STAGING_FIELDS) + list(SC_STAGING_FIELDS),
            num_samples=64,
            consumer_tasks=["finalize"],
        )
        client.register_partition(
            partition_id=CANONICAL,
            fields=[
                "input_ids", "input_lengths", "generation_logprobs", "token_mask",
                "sample_mask", "prompt_ids_for_adv", "total_reward", "mask_sample",
                "truncated", *SC_FIELDS,
            ],
            num_samples=64,
            consumer_tasks=["train"],
        )
        records, receipt, row, expected = _stage(client, "live_g0", m=M, with_draws_for=("c1",))
        finalizer = _finalizer(client, M)
        finalized = finalizer.finalize_group(
            "live", ["live_g0"], [receipt], [1.0], mask_sample=[False],
            fallback_weight_version=0, prompt_idx=0,
        )
        assert not finalized.dropped and finalized.valid_row_count == 1
        rows = client.get_samples(
            sample_ids=["live_g0"], partition_id=CANONICAL, select_fields=list(SC_FIELDS)
        )
        exp_ids, exp_lps, exp_mask = _expected_row(row, expected)
        got_ids = torch.as_tensor(rows[SC_SAMPLE_TOKEN_IDS_FIELD][0])
        got_mask = torch.as_tensor(rows[SC_SAMPLE_MASK_FIELD][0]).flatten()
        assert torch.equal(got_ids.reshape(exp_ids.shape), exp_ids)
        assert torch.equal(got_mask, exp_mask)
