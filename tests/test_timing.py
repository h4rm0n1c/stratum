import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from stratum.pipeline import StratumPipeline
from stratum.stage import DeviceStage
from stratum.timing import TimingRecorder


class _Prefix(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        return (hidden, None, input_ids, None, {}, labels, 0)


class _Layer(nn.Module):
    def forward(self, input_data):
        hidden = input_data[0] + 1.0
        return (hidden,) + input_data[1:]


class _Postfix(nn.Module):
    def forward(self, input_data):
        return SimpleNamespace(loss=input_data[0].sum())


class TimingRecorderTest(unittest.TestCase):
    def test_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timing.jsonl"
            recorder = TimingRecorder(path, use_cuda_events=False)
            with recorder.span("unit", stage=1):
                _ = sum(range(10))
            recorder.close()

            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "timing")
        self.assertEqual(rows[0]["name"], "unit")
        self.assertEqual(rows[0]["stage"], 1)
        self.assertGreaterEqual(rows[0]["wall_ms"], 0.0)

    def test_pipeline_records_expected_spans(self):
        pipeline = StratumPipeline(
            _Prefix(),
            [DeviceStage([_Layer()], device_id=0)],
            _Postfix(),
        )
        recorder = TimingRecorder(use_cuda_events=False)
        pipeline.set_timing_recorder(recorder)

        output = pipeline(
            torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            labels=torch.tensor([[1, 2]]),
        )
        pipeline.free_all_weights()

        names = [record["name"] for record in recorder.records]
        self.assertEqual(float(output.loss.detach()), 5.0)
        self.assertLess(names.index("prefix_upload"), names.index("prefix_forward"))
        self.assertIn("stage_upload", names)
        self.assertIn("stage_forward", names)
        self.assertIn("postfix_upload", names)
        self.assertIn("postfix_forward", names)
        self.assertIn("stage_free", names)


if __name__ == "__main__":
    unittest.main()
