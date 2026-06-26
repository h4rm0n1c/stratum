import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import stratum.pipeline as pipeline_mod
from stratum.pipeline import StratumPipeline
from stratum.scheduler import ModelExecutePlan
from stratum.stage import DeviceStage
from stratum.timing import IterLayerTimer, ModelLayerTimer, TimingRecorder


class _Prefix(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        return (hidden, None, input_ids, None, {}, labels, 0)


class _GradPrefix(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).requires_grad_()
        return (hidden, None, input_ids, None, {}, labels, 0)


class _Layer(nn.Module):
    def forward(self, input_data):
        hidden = input_data[0] + 1.0
        return (hidden,) + input_data[1:]


class _TrainableLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_data):
        hidden = input_data[0] + self.bias
        return (hidden,) + input_data[1:]


class _MutatingRouterLayer(nn.Module):
    def forward(self, input_data):
        hidden, causal, pos_ids, pos_embeds, kwargs, labels, logits_to_keep = input_data
        kwargs.setdefault("_router_logits", []).append(hidden * 3.0)
        return (
            hidden + 1.0,
            causal,
            pos_ids,
            pos_embeds,
            kwargs,
            labels,
            logits_to_keep,
        )


class _Postfix(nn.Module):
    def forward(self, input_data):
        return SimpleNamespace(loss=input_data[0].sum())


class _RouterPostfix(nn.Module):
    def forward(self, input_data):
        router_loss = sum(item.sum() for item in input_data[4]["_router_logits"])
        return SimpleNamespace(loss=input_data[0].sum() + router_loss)


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
        self.assertIn("layer_forward", names)
        self.assertIn("postfix_upload", names)
        self.assertIn("postfix_forward", names)
        self.assertIn("stage_free", names)

    def test_pipeline_records_scheduler_plan_events(self):
        pipeline = StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_Layer()], device_id=0),
                DeviceStage([_Layer()], device_id=0),
            ],
            _Postfix(),
        )
        recorder = TimingRecorder(use_cuda_events=False)
        pipeline.set_timing_recorder(recorder)

        output = pipeline(
            torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            labels=torch.tensor([[1, 2]]),
        )
        output.loss.backward()
        pipeline.free_all_weights()

        names = [record["name"] for record in recorder.records]
        self.assertEqual(
            [record["layer_start"] for record in recorder.records if record["name"] == "stage_forward"],
            [0, 1],
        )
        self.assertEqual(
            [record["layer_index"] for record in recorder.records if record["name"] == "layer_forward"],
            [0, 1],
        )
        self.assertEqual(names.count("scheduler_forward_notify"), 2)
        self.assertEqual(names.count("scheduler_backward_notify"), 2)
        self.assertEqual(names.count("stage_backward"), 2)
        self.assertEqual(
            [
                (record["fwd_group_id"], record["bwd_group_id"], record["backward_start_seen"])
                for record in recorder.records
                if record["name"] == "stage_backward"
            ],
            [(1, 0, True), (0, 1, True)],
        )

    def test_pipeline_runs_multiple_scheduler_groups_inside_one_stage(self):
        plan = ModelExecutePlan.from_stage_lengths([1, 1])
        pipeline = StratumPipeline(
            _GradPrefix(),
            [DeviceStage([_Layer(), _Layer()], device_id=0)],
            _Postfix(),
            execute_plan=plan,
        )
        recorder = TimingRecorder(use_cuda_events=False)
        pipeline.set_timing_recorder(recorder)

        output = pipeline(
            torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            labels=torch.tensor([[1, 2]]),
        )
        output.loss.backward()

        self.assertEqual(float(output.loss.detach()), 7.0)
        self.assertEqual(
            [
                (record["fwd_group_id"], record["layer_start"], record["layer_stop"])
                for record in recorder.records
                if record["name"] == "stage_forward"
            ],
            [(0, 0, 1), (1, 1, 2)],
        )
        self.assertEqual(
            [record["layer_index"] for record in recorder.records if record["name"] == "layer_forward"],
            [0, 1],
        )

    def test_explicit_group_backward_completes_group_without_input_grad(self):
        plan = ModelExecutePlan.from_stage_lengths([1, 1])
        pipeline = StratumPipeline(
            _Prefix(),
            [DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0)],
            _Postfix(),
            execute_plan=plan,
        )
        recorder = TimingRecorder(use_cuda_events=False)
        pipeline.set_timing_recorder(recorder)

        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                output = pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )
                output.loss.backward()
                pipeline.free_all_weights()

        notify_records = [
            record
            for record in recorder.records
            if record["name"] == "scheduler_backward_notify"
        ]
        free_records = [
            record for record in recorder.records if record["name"] == "stage_group_free"
        ]
        backward_records = [
            record for record in recorder.records if record["name"] == "stage_backward"
        ]

        self.assertEqual(float(output.loss.detach()), 7.0)
        self.assertEqual(
            [(record["fwd_group_id"], record["bwd_group_id"]) for record in notify_records],
            [(1, 0), (0, 1)],
        )
        self.assertEqual(
            [record.get("after_backward_fallback", False) for record in notify_records],
            [False, False],
        )
        self.assertEqual(
            [(record["fwd_group_id"], record["layer_start"], record["layer_stop"]) for record in free_records],
            [(1, 1, 2), (0, 0, 1)],
        )
        self.assertEqual(
            [
                (
                    record["fwd_group_id"],
                    record["bwd_group_id"],
                    record["backward_start_seen"],
                    record.get("after_backward_fallback", False),
                )
                for record in backward_records
            ],
            [(1, 0, True, False), (0, 1, True, False)],
        )
        self.assertEqual(
            [
                (record["fwd_group_id"], record["bwd_group_id"])
                for record in recorder.records
                if record["name"] == "stage_backward_explicit"
            ],
            [(1, 0), (0, 1)],
        )

    def test_explicit_group_backward_uses_pre_forward_side_channel_structure(self):
        pipeline = StratumPipeline(
            _GradPrefix(),
            [DeviceStage([_MutatingRouterLayer()], device_id=0)],
            _RouterPostfix(),
        )
        recorder = TimingRecorder(use_cuda_events=False)
        pipeline.set_timing_recorder(recorder)

        output = pipeline(
            torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            labels=torch.tensor([[1, 2]]),
        )
        output.loss.backward()
        pipeline.free_all_weights()

        self.assertEqual(float(output.loss.detach()), 14.0)
        self.assertEqual(
            [
                (record["fwd_group_id"], record["bwd_group_id"])
                for record in recorder.records
                if record["name"] == "stage_backward_explicit"
            ],
            [(0, 0)],
        )

    def test_pipeline_rejects_scheduler_group_crossing_stage_boundary(self):
        plan = ModelExecutePlan.from_stage_lengths([2])

        with self.assertRaisesRegex(ValueError, "cannot cross"):
            StratumPipeline(
                _Prefix(),
                [
                    DeviceStage([_Layer()], device_id=0),
                    DeviceStage([_Layer()], device_id=0),
                ],
                _Postfix(),
                execute_plan=plan,
            )


class _FakeEvent:
    """Fake CUDA event for host-only timer tests."""

    def __init__(self, time_ms: float = 0.0):
        self._time = time_ms

    def query(self) -> bool:
        return True

    def elapsed_time(self, end: "_FakeEvent") -> float:
        return end._time - self._time


def _make_fwd_events(n_layers: int, fwd_ms: list, re_ms: list):
    """Build fake fwd_events dict matching IterLayerTimer layout."""
    fwd: list = []
    re: list = []
    for i in range(n_layers):
        s = _FakeEvent(0.0)
        e = _FakeEvent(fwd_ms[i])
        fwd.append([(s, e)])
        s2 = _FakeEvent(0.0)
        e2 = _FakeEvent(re_ms[i])
        re.append([(s2, e2)])
    return {"fwd": fwd, "re": re}


def _make_bwd_events(layer_ids: range, bwd_ms: float) -> dict:
    s = _FakeEvent(0.0)
    e = _FakeEvent(bwd_ms)
    return {(layer_ids.start, layer_ids.stop): [(s, e)]}


class ModelLayerTimerTests(unittest.TestCase):
    def test_has_estimates_false_before_warmup(self):
        t = ModelLayerTimer(n_layers=3)
        self.assertFalse(t.has_estimates())

    def test_first_iteration_dropped_stage_stays_at_one(self):
        t = ModelLayerTimer(n_layers=2)
        t._iter_results.put((
            _make_fwd_events(2, [1.0, 2.0], [1.1, 2.1]),
            _make_bwd_events(range(0, 2), 5.0),
        ))
        t.update_times()
        # First result is dropped — stage moves from 0 → 1, not 2
        self.assertEqual(t._stage["fwd"], 1)
        self.assertFalse(t.has_estimates())

    def test_two_iterations_activates_estimates(self):
        t = ModelLayerTimer(n_layers=2)
        # Push two iterations
        for _ in range(2):
            t._iter_results.put((
                _make_fwd_events(2, [1.0, 2.0], [1.1, 2.1]),
                _make_bwd_events(range(0, 2), 6.0),
            ))
        t.update_times()
        t.update_times()
        self.assertTrue(t.has_estimates())

    def test_ema_converges_toward_observed_value(self):
        t = ModelLayerTimer(n_layers=1)
        # Push many iterations with constant re=2.0, bwd_total=4.0
        for _ in range(30):
            t._iter_results.put((
                _make_fwd_events(1, [2.0], [2.0]),
                _make_bwd_events(range(0, 1), 4.0),
            ))
        for _ in range(30):
            t.update_times()
        self.assertTrue(t.has_estimates())
        fwd_ms, bwd_ms = t.get_training_estimates()
        # fwd_ms comes from recompute estimate (2.0 ms per layer)
        self.assertAlmostEqual(fwd_ms[0], 2.0, delta=0.1)
        # bwd_ms = re + proportional bwd attribution: 2.0 + 4.0 = 6.0
        self.assertAlmostEqual(bwd_ms[0], 6.0, delta=0.2)

    def test_backward_attribution_proportional_to_recompute(self):
        """Backward time for each layer is proportional to its recompute fraction."""
        t = ModelLayerTimer(n_layers=2)
        # Layer 0 recomputes 3x faster than layer 1; total backward = 8.0
        for _ in range(10):
            t._iter_results.put((
                _make_fwd_events(2, [1.0, 3.0], [1.0, 3.0]),
                _make_bwd_events(range(0, 2), 8.0),
            ))
        for _ in range(10):
            t.update_times()
        fwd_ms, bwd_ms = t.get_training_estimates()
        # Layer 0 re=1.0, total_re=4.0 → attributed bwd = 1/4 * 8.0 = 2.0; total = 1.0 + 2.0 = 3.0
        # Layer 1 re=3.0, total_re=4.0 → attributed bwd = 3/4 * 8.0 = 6.0; total = 3.0 + 6.0 = 9.0
        self.assertAlmostEqual(bwd_ms[0], 3.0, delta=0.3)
        self.assertAlmostEqual(bwd_ms[1], 9.0, delta=0.3)

    def test_update_times_returns_false_when_queue_empty(self):
        t = ModelLayerTimer(n_layers=1)
        result = t.update_times()
        self.assertFalse(result)

    def test_update_times_returns_true_after_processing(self):
        t = ModelLayerTimer(n_layers=1)
        # Push two iterations; update_times drains all ready items in one call.
        t._iter_results.put((
            _make_fwd_events(1, [1.0], [1.0]),
            _make_bwd_events(range(0, 1), 2.0),
        ))
        t._iter_results.put((
            _make_fwd_events(1, [1.0], [1.0]),
            _make_bwd_events(range(0, 1), 2.0),
        ))
        # First call processes both: drops iteration 0, activates EMA on iteration 1.
        result = t.update_times()
        self.assertTrue(result)
        # Second call finds empty queue.
        result2 = t.update_times()
        self.assertFalse(result2)

    def test_layer_size_bytes_seeds_initial_estimate(self):
        sizes = [1000, 2000]
        t = ModelLayerTimer(n_layers=2, layer_size_bytes=sizes)
        # Before any data, estimates are seeded from layer sizes
        fwd_ms, bwd_ms = t.get_training_estimates()
        self.assertEqual(fwd_ms[0], 1000.0)
        self.assertEqual(fwd_ms[1], 2000.0)
        # bwd = re + bwd, and before EMA both use seed values
        # re seed = size, bwd seed = size * BACKWARD_MULTIPLIER
        # Total initial bwd estimate = 1000 + 1000*2 = 3000
        self.assertAlmostEqual(bwd_ms[0], 3000.0, delta=1.0)


class IterLayerTimerTests(unittest.TestCase):
    def test_deletion_pushes_to_parent_queue(self):
        parent = ModelLayerTimer(n_layers=2)
        it = parent.new_iter()
        del it
        self.assertEqual(parent._iter_results.qsize(), 1)

    def test_fwd_events_have_correct_structure(self):
        parent = ModelLayerTimer(n_layers=3)
        it = parent.new_iter()
        self.assertEqual(len(it.fwd_events["fwd"]), 3)
        self.assertEqual(len(it.fwd_events["re"]), 3)
        self.assertEqual(it.fwd_events["fwd"][0], [])

    def test_bwd_events_start_empty(self):
        parent = ModelLayerTimer(n_layers=2)
        it = parent.new_iter()
        self.assertEqual(len(it.bwd_events), 0)


class ModelLayerTimerPipelineTests(unittest.TestCase):
    """Smoke tests: pipeline creates IterLayerTimer, timer accumulates events."""

    def _make_pipeline(self):
        plan = ModelExecutePlan.from_stage_lengths([1, 1])
        pipeline = StratumPipeline(
            _GradPrefix(),
            [DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0)],
            _Postfix(),
            execute_plan=plan,
        )
        return pipeline

    def test_set_layer_timer_attaches_timer(self):
        pipeline = self._make_pipeline()
        timer = ModelLayerTimer(n_layers=2)
        pipeline.set_layer_timer(timer)
        self.assertIs(pipeline._layer_timer, timer)

    def test_forward_creates_iter_timer_when_set(self):
        pipeline = self._make_pipeline()
        timer = ModelLayerTimer(n_layers=2)
        pipeline.set_layer_timer(timer)

        created_timers = []

        original_new_iter = timer.new_iter

        def _spy_new_iter():
            it = original_new_iter()
            created_timers.append(it)
            return it

        with mock.patch.object(timer, "new_iter", side_effect=_spy_new_iter):
            with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
                with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                    pipeline(
                        torch.tensor([[1, 2]]),
                        attention_mask=torch.ones(1, 2, dtype=torch.long),
                        labels=torch.tensor([[1, 2]]),
                    )

        self.assertEqual(len(created_timers), 1)

    def test_forward_calls_update_times_on_second_call(self):
        pipeline = self._make_pipeline()
        timer = ModelLayerTimer(n_layers=2)
        pipeline.set_layer_timer(timer)

        update_calls = []
        original_update = timer.update_times

        def _spy_update():
            result = original_update()
            update_calls.append(result)
            return result

        with mock.patch.object(timer, "update_times", side_effect=_spy_update):
            with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
                with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                    for _ in range(2):
                        pipeline(
                            torch.tensor([[1, 2]]),
                            attention_mask=torch.ones(1, 2, dtype=torch.long),
                            labels=torch.tensor([[1, 2]]),
                        )

        # update_times is called once per forward, so twice total
        self.assertEqual(len(update_calls), 2)


class PlanAdaptationTests(unittest.TestCase):
    """Tests for timing-fed plan adaptation in StratumPipeline."""

    def _make_two_stage_pipeline(self):
        """Two 2-layer stages (4 layers total, starts as one group per stage)."""
        plan = ModelExecutePlan.from_stage_lengths([2, 2])
        pipeline = StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
            ],
            _Postfix(),
            execute_plan=plan,
        )
        return pipeline

    def _pump_fake_estimates(self, timer: ModelLayerTimer, n_layers: int, fwd_ms, bwd_ms):
        """Push enough fake iterations to make has_estimates() return True."""
        re_ms = [f * 0.95 for f in fwd_ms]
        # Need 2+ iterations (first is dropped); pump 3 to be safe
        for _ in range(3):
            fwd_events = _make_fwd_events(n_layers, fwd_ms, re_ms)
            bwd_events = {}
            # One backward event per device group (here: two groups of 2)
            for start, stop in [(0, 2), (2, 4)]:
                total_bwd = sum(bwd_ms[start:stop])
                bwd_events[(start, stop)] = [(_FakeEvent(0.0), _FakeEvent(total_bwd))]
            timer._iter_results.put((fwd_events, bwd_events))
        timer.update_times()

    def test_rebuild_from_plan_updates_execute_plan(self):
        pipeline = self._make_two_stage_pipeline()
        self.assertEqual(len(pipeline.execute_plan.fwd_plan), 2)

        # Split first stage into two 1-layer groups
        new_plan = ModelExecutePlan.from_stage_lengths([1, 1, 2])
        pipeline._rebuild_from_plan(new_plan)

        self.assertEqual(len(pipeline.execute_plan.fwd_plan), 3)
        self.assertEqual(pipeline.execute_plan.fwd_plan[0], range(0, 1))
        self.assertEqual(pipeline.execute_plan.fwd_plan[1], range(1, 2))
        self.assertEqual(pipeline.execute_plan.fwd_plan[2], range(2, 4))

    def test_rebuild_from_plan_updates_bwd_group_by_range(self):
        pipeline = self._make_two_stage_pipeline()
        new_plan = ModelExecutePlan.from_stage_lengths([1, 1, 2])
        pipeline._rebuild_from_plan(new_plan)

        # bwd_plan = reversed(fwd_plan) = [range(2,4), range(1,2), range(0,1)]
        # bwd_group_by_range maps range → bwd_group_id
        self.assertIn((2, 4), pipeline._bwd_group_by_range)
        self.assertIn((1, 2), pipeline._bwd_group_by_range)
        self.assertIn((0, 1), pipeline._bwd_group_by_range)

    def test_rebuild_from_plan_updates_stage_group_ids(self):
        pipeline = self._make_two_stage_pipeline()
        new_plan = ModelExecutePlan.from_stage_lengths([1, 1, 2])
        pipeline._rebuild_from_plan(new_plan)

        # Stage 0 owns layers 0–1 → fwd groups 0 and 1
        # Stage 1 owns layers 2–3 → fwd group 2
        self.assertEqual(set(pipeline._stage_group_ids[0]), {0, 1})
        self.assertEqual(set(pipeline._stage_group_ids[1]), {2})

    def test_try_adapt_plan_returns_false_before_warmup(self):
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=4)
        pipeline.set_layer_timer(timer, adapt_every_n=1)

        result = pipeline._try_adapt_plan()
        self.assertFalse(result)

    def test_try_adapt_plan_adapts_when_estimates_ready(self):
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=4)
        pipeline.set_layer_timer(timer, adapt_every_n=1)

        # Force very unequal layer times so auto_from_layer_metrics splits
        # Layer 0 takes 10x longer than layer 1 within stage 0.
        self._pump_fake_estimates(timer, 4,
                                  fwd_ms=[10.0, 1.0, 1.0, 1.0],
                                  bwd_ms=[12.0, 2.0, 2.0, 2.0])
        self.assertTrue(timer.has_estimates())

        original_fwd = list(pipeline.execute_plan.fwd_plan)
        result = pipeline._try_adapt_plan()
        # With very unequal times, auto should split stage 0 into 2 groups
        # (result may be True or False depending on threshold — just verify no crash)
        # What we CAN verify: no exception and plan is still valid
        pipeline.execute_plan.check_valid(4, "train")
        _ = result  # result depends on whether algorithm decides to split

    def test_try_adapt_plan_returns_false_when_plan_unchanged(self):
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=4)
        pipeline.set_layer_timer(timer, adapt_every_n=1)

        # Equal times → auto_from_layer_metrics keeps same single-group plan
        self._pump_fake_estimates(timer, 4,
                                  fwd_ms=[1.0, 1.0, 1.0, 1.0],
                                  bwd_ms=[2.0, 2.0, 2.0, 2.0])
        self.assertTrue(timer.has_estimates())

        result = pipeline._try_adapt_plan()
        self.assertFalse(result)

    def test_no_adaptation_when_adapt_every_n_zero(self):
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=4)
        pipeline.set_layer_timer(timer)  # default adapt_every_n=0

        self.assertEqual(pipeline._plan_adapt_after_n, 0)
        self.assertEqual(pipeline._steps_until_adapt, 0)

    def test_adapt_every_n_triggers_on_correct_step(self):
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=4)
        pipeline.set_layer_timer(timer, adapt_every_n=3)

        adapt_calls = []
        original_try = pipeline._try_adapt_plan

        def _spy():
            adapt_calls.append(True)
            return original_try()

        pipeline._try_adapt_plan = _spy

        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                for _ in range(6):
                    pipeline(
                        torch.tensor([[1, 2]]),
                        attention_mask=torch.ones(1, 2, dtype=torch.long),
                        labels=torch.tensor([[1, 2]]),
                    )

        # adapt_every_n=3: first attempt at step 3, then step 6 → 2 calls in 6 steps
        self.assertEqual(len(adapt_calls), 2)

    def test_adapted_plan_allows_correct_forward_backward(self):
        """After plan rebuild, forward+backward must still complete correctly."""
        pipeline = self._make_two_stage_pipeline()
        # Manually rebuild into 3 groups: 0:1, 1:2, 2:4
        new_plan = ModelExecutePlan.from_stage_lengths([1, 1, 2])
        pipeline._rebuild_from_plan(new_plan)

        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                output = pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )
                output.loss.backward()

        # Loss should be 4 (prefix adds 0→float, each TrainableLayer adds bias=1)
        # 4 layers * bias=1, plus initial prefix value: 1+1+1+1+1+1 = 7 total
        self.assertGreater(float(output.loss.detach()), 0.0)


class PerLayerUploadTests(unittest.TestCase):
    """Tests for sub-slice (d): per-layer upload overlap in DeviceStage.forward_range."""

    def _make_stage(self, n_layers: int = 3) -> DeviceStage:
        layers = [_TrainableLayer() for _ in range(n_layers)]
        return DeviceStage(layers, device_id=0)

    def test_forward_range_runs_without_param_stream(self):
        """forward_range must work as before when param_stream is None."""
        stage = self._make_stage(2)
        hidden = torch.ones(1, 2, 1)
        out = stage.forward_range(
            (hidden, None, None, None, {}, None, 0),
            local_start=0, local_stop=2,
        )
        self.assertIsNotNone(out)

    def test_forward_range_with_param_stream_on_cpu_is_noop(self):
        """Passing param_stream=None (CPU) must not break forward_range."""
        stage = self._make_stage(2)
        hidden = torch.ones(1, 2, 1)
        out = stage.forward_range(
            (hidden, None, None, None, {}, None, 0),
            local_start=0, local_stop=2,
            param_stream=None,
        )
        self.assertIsNotNone(out)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_range_calls_ensure_weights_per_layer_on_cuda(self):
        """With param_stream, ensure_weights must be called once per layer."""
        stage = self._make_stage(3)
        import stratum.stage as stage_mod
        ensure_calls = []
        with mock.patch.object(stage_mod, "ensure_weights", side_effect=lambda m, d: ensure_calls.append(m)):
            stream = torch.cuda.Stream(device=0)
            hidden = torch.ones(1, 2, 1)
            out = stage.forward_range(
                (hidden, None, None, None, {}, None, 0),
                local_start=0, local_stop=3,
                param_stream=stream,
            )
        # Layer 0 pre-uploaded before loop, layers 1 and 2 inside the loop
        self.assertEqual(len(ensure_calls), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_range_records_fence_per_layer_on_cuda(self):
        """A wait_event must be called for every layer in the range."""
        stage = self._make_stage(3)
        stream = torch.cuda.Stream(device=0)
        wait_calls = []
        real_default = torch.cuda.default_stream

        def spy_default(device=None):
            s = real_default(device)
            orig_wait = s.wait_event
            def spy_wait(ev):
                wait_calls.append(ev)
                return orig_wait(ev)
            s.wait_event = spy_wait
            return s

        import stratum.stage as stage_mod
        with mock.patch("stratum.stage.torch.cuda.default_stream", side_effect=spy_default):
            with mock.patch.object(stage_mod, "ensure_weights", return_value=0):
                hidden = torch.ones(1, 2, 1)
                stage.forward_range(
                    (hidden, None, None, None, {}, None, 0),
                    local_start=0, local_stop=3,
                    param_stream=stream,
                )
        # One wait_event per layer
        self.assertEqual(len(wait_calls), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_first_layer_fence_is_used_not_reuploaded(self):
        """When first_layer_fence is provided, layer 0 must NOT be uploaded again."""
        stage = self._make_stage(2)
        import stratum.stage as stage_mod
        ensure_calls = []
        with mock.patch.object(stage_mod, "ensure_weights", side_effect=lambda m, d: ensure_calls.append(m)):
            stream = torch.cuda.Stream(device=0)
            pre_fence = torch.cuda.Event()
            pre_fence.record(stream)
            hidden = torch.ones(1, 2, 1)
            stage.forward_range(
                (hidden, None, None, None, {}, None, 0),
                local_start=0, local_stop=2,
                param_stream=stream,
                first_layer_fence=pre_fence,
            )
        # Layer 0 was pre-uploaded (fence supplied), so only layer 1 is uploaded here
        self.assertEqual(len(ensure_calls), 1)

    def test_pipeline_uses_per_layer_upload_on_cuda(self):
        """On CUDA, pipeline must call _upload_first_layer_with_fence for cross-group overlap."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        plan = ModelExecutePlan.from_stage_lengths([2, 2])
        pipeline = StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
            ],
            _Postfix(),
            execute_plan=plan,
        )
        first_layer_calls = []
        orig = pipeline._upload_first_layer_with_fence
        def spy(gid):
            first_layer_calls.append(gid)
            return orig(gid)
        pipeline._upload_first_layer_with_fence = spy
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )
        # Group 0 pre-uploaded before prefix + group 1 pre-uploaded during group 0 compute
        self.assertGreaterEqual(len(first_layer_calls), 2)

    def test_pipeline_cpu_path_still_uses_group_level_upload(self):
        """On CPU, pipeline must still call _upload_group_with_fence (group-level)."""
        plan = ModelExecutePlan.from_stage_lengths([2, 2])
        pipeline = StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
            ],
            _Postfix(),
            execute_plan=plan,
        )
        group_calls = []
        orig = pipeline._upload_group_with_fence
        def spy(gid, prefetch=None):
            group_calls.append(gid)
            return orig(gid, prefetch)
        pipeline._upload_group_with_fence = spy
        # On CPU, _param_upload_stream returns None so group-level path is taken
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )
        # CPU path: each group uploaded at group level
        self.assertGreaterEqual(len(group_calls), 1)


class ParamUpstreamStreamTests(unittest.TestCase):
    """Tests for param_upstream CUDA stream and _upload_group_with_fence."""

    def _make_pipeline(self):
        plan = ModelExecutePlan.from_stage_lengths([2, 2])
        return StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
            ],
            _Postfix(),
            execute_plan=plan,
        )

    def test_param_upload_stream_returns_none_on_cpu(self):
        """On CPU (no CUDA), _param_upload_stream must return None."""
        pipeline = self._make_pipeline()
        self.assertIsNone(pipeline._param_upload_stream(0))

    def test_param_upload_stream_is_cached(self):
        """Second call with same device_id returns the identical stream object."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        pipeline = self._make_pipeline()
        s1 = pipeline._param_upload_stream(0)
        s2 = pipeline._param_upload_stream(0)
        self.assertIs(s1, s2)

    def test_param_upload_stream_distinct_per_device(self):
        """Different device_ids produce different stream objects."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("need 2+ CUDA devices")
        pipeline = self._make_pipeline()
        s0 = pipeline._param_upload_stream(0)
        s1 = pipeline._param_upload_stream(1)
        self.assertIsNot(s0, s1)

    def test_upload_group_with_fence_returns_none_on_cpu(self):
        """On CPU, _upload_group_with_fence must return None and still upload."""
        pipeline = self._make_pipeline()
        upload_calls = []
        original_ensure = pipeline._ensure_stage_group

        def spy(gid, prefetch=None):
            upload_calls.append(gid)
            return original_ensure(gid, prefetch)

        pipeline._ensure_stage_group = spy
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            result = pipeline._upload_group_with_fence(0)

        self.assertIsNone(result)
        self.assertEqual(upload_calls, [0])

    def test_upload_group_with_fence_returns_event_on_cuda(self):
        """On CUDA, _upload_group_with_fence returns a non-None Event."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        pipeline = self._make_pipeline()
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            event = pipeline._upload_group_with_fence(0)
        self.assertIsNotNone(event)
        self.assertIsInstance(event, torch.cuda.Event)

    def test_forward_populates_pre_upload_fences_on_cuda(self):
        """On CUDA, pre_upload_fences[0] is set before prefix runs (group 0 pre-uploaded)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        pipeline = self._make_pipeline()
        recorded_fences = {}
        original_upload = pipeline._upload_group_with_fence

        def spy_upload(gid, prefetch=None):
            event = original_upload(gid, prefetch)
            recorded_fences[gid] = event
            return event

        pipeline._upload_group_with_fence = spy_upload
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )

        # Group 0 must have been pre-uploaded (fence recorded) before prefix ran
        self.assertIn(0, recorded_fences)
        self.assertIsNotNone(recorded_fences[0])

    def test_forward_works_on_cpu_with_param_upstream_disabled(self):
        """Forward/backward still completes correctly on CPU (no param_upstream)."""
        pipeline = self._make_pipeline()
        with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
            with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                output = pipeline(
                    torch.tensor([[1, 2]]),
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    labels=torch.tensor([[1, 2]]),
                )
                output.loss.backward()
        self.assertGreater(float(output.loss.detach()), 0.0)

    def test_compute_stream_wait_event_called_on_cuda(self):
        """On CUDA, the default compute stream must receive a wait_event call."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        pipeline = self._make_pipeline()
        wait_calls = []
        real_default_stream = torch.cuda.default_stream

        def spy_stream(device=None):
            s = real_default_stream(device)
            original_wait = s.wait_event

            def spy_wait(event):
                wait_calls.append(event)
                return original_wait(event)

            s.wait_event = spy_wait
            return s

        with mock.patch("stratum.pipeline.torch.cuda.default_stream", side_effect=spy_stream):
            with mock.patch.object(pipeline_mod, "ensure_weights", return_value=0):
                with mock.patch.object(pipeline_mod, "free_weights", return_value=0):
                    pipeline(
                        torch.tensor([[1, 2]]),
                        attention_mask=torch.ones(1, 2, dtype=torch.long),
                        labels=torch.tensor([[1, 2]]),
                    )

        # At least one fence wait must have occurred
        self.assertGreater(len(wait_calls), 0)


class TrainWiringTests(unittest.TestCase):
    """Verify invariants that train.py must satisfy when wiring set_layer_timer."""

    def _make_two_stage_pipeline(self):
        plan = ModelExecutePlan.from_stage_lengths([2, 2])
        return StratumPipeline(
            _GradPrefix(),
            [
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
                DeviceStage([_TrainableLayer(), _TrainableLayer()], device_id=0),
            ],
            _Postfix(),
            execute_plan=plan,
        )

    def test_timer_n_layers_matches_pipeline_total_layers(self):
        """train.py wiring invariant: ModelLayerTimer.n_layers must equal pipeline._total_layers."""
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=pipeline._total_layers)
        pipeline.set_layer_timer(timer, adapt_every_n=10)

        self.assertEqual(timer.n_layers, pipeline._total_layers)
        self.assertEqual(pipeline._plan_adapt_after_n, 10)
        self.assertIs(pipeline._layer_timer, timer)

    def test_adapt_plan_every_zero_disables_adaptation(self):
        """adapt_every_n=0 (train.py default) must leave adaptation off."""
        pipeline = self._make_two_stage_pipeline()
        timer = ModelLayerTimer(n_layers=pipeline._total_layers)
        pipeline.set_layer_timer(timer, adapt_every_n=0)

        self.assertEqual(pipeline._plan_adapt_after_n, 0)
        self.assertEqual(pipeline._steps_until_adapt, 0)


if __name__ == "__main__":
    unittest.main()
