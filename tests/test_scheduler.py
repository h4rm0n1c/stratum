import threading
import unittest

import torch

from stratum.scheduler import (
    BackwardScheduleSimulator,
    ModelExecutePlan,
    ModelTracker,
    chunk_layer_params,
)


class ModelExecutePlanTest(unittest.TestCase):
    def test_stage_lengths_build_train_plan(self):
        plan = ModelExecutePlan.from_stage_lengths([2, 1, 3])

        self.assertEqual([list(r) for r in plan.fwd_plan], [[0, 1], [2], [3, 4, 5]])
        self.assertEqual([list(r) for r in plan.bwd_plan], [[3, 4, 5], [2], [0, 1]])
        plan.check_valid(6, "train")

    def test_check_valid_rejects_gap(self):
        plan = ModelExecutePlan(fwd_plan=[range(0, 1), range(2, 3)])

        with self.assertRaisesRegex(ValueError, "Specify 2 after 0"):
            plan.check_valid(3, "infer")

    def test_auto_from_layer_metrics_returns_valid_train_plan(self):
        plan = ModelExecutePlan.auto_from_layer_metrics(
            "train",
            fwd_times=[1.0, 1.0, 2.0, 1.0],
            bwd_times=[1.0, 2.0, 1.0, 1.0],
            layer_fwd_size_gib=[0.4, 0.4, 0.4, 0.4],
            layer_bwd_size_gib=[0.6, 0.6, 0.6, 0.6],
            min_stages=2,
            upper_threshold=2.0,
            model_memory_limit_gib=2.0,
        )

        plan.check_valid(4, "train")
        self.assertGreaterEqual(len(plan.fwd_plan), 1)
        self.assertGreaterEqual(len(plan.bwd_plan), 1)

    def test_auto_from_layer_metrics_supports_infer(self):
        plan = ModelExecutePlan.auto_from_layer_metrics(
            "infer",
            fwd_times=[1.0, 1.0, 1.0],
            layer_fwd_size_gib=[0.1, 0.1, 0.1],
            model_memory_limit_gib=1.0,
        )

        plan.check_valid(3, "infer")
        self.assertEqual(plan.bwd_plan, [])


class ModelTrackerTest(unittest.TestCase):
    def test_forward_and_backward_waits_are_notified_by_stage(self):
        tracker = ModelTracker(ModelExecutePlan.from_stage_lengths([1, 1]))
        events: list[str] = []

        def wait_forward():
            tracker.forward_wait_for(0)
            events.append("forward")

        def wait_backward():
            tracker.backward_wait_for(0)
            events.append("backward")

        fwd_thread = threading.Thread(target=wait_forward)
        bwd_thread = threading.Thread(target=wait_backward)
        fwd_thread.start()
        bwd_thread.start()

        tracker.forward_notify(0)
        tracker.backward_notify(0)
        fwd_thread.join(timeout=1.0)
        bwd_thread.join(timeout=1.0)

        self.assertFalse(fwd_thread.is_alive())
        self.assertFalse(bwd_thread.is_alive())
        self.assertCountEqual(events, ["forward", "backward"])

    def test_backward_need_input_matches_stage_starts(self):
        tracker = ModelTracker(ModelExecutePlan.from_stage_lengths([2, 2]))

        self.assertTrue(tracker.backward_need_input(2))
        self.assertTrue(tracker.backward_need_input(0))
        self.assertFalse(tracker.backward_need_input(1))


class BackwardScheduleSimulatorTest(unittest.TestCase):
    def test_rotates_and_resets_tags(self):
        simulator = BackwardScheduleSimulator(n_devices=2)

        first = simulator.get_next_tag()
        new = first + torch.tensor(1.0)
        simulator.update_current_tag(new)
        simulator.get_next_tag()
        self.assertIs(simulator.get_next_tag(), new)

        simulator.reset()
        self.assertIsNot(simulator.get_next_tag(), new)


class ChunkLayerParamsTest(unittest.TestCase):
    def test_balances_largest_tensors_across_chunks(self):
        pairs = [
            (torch.empty(8, dtype=torch.float32), torch.empty(8, dtype=torch.float32)),
            (torch.empty(4, dtype=torch.float32), torch.empty(4, dtype=torch.float32)),
            (torch.empty(4, dtype=torch.float32), torch.empty(4, dtype=torch.float32)),
            (torch.empty(2, dtype=torch.float32), torch.empty(2, dtype=torch.float32)),
        ]

        chunks = chunk_layer_params(pairs, 2)
        totals = [
            sum(src.numel() * src.element_size() for src, _ in chunk)
            for chunk in chunks
        ]

        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(max(totals) - min(totals), 8 * 4)


if __name__ == "__main__":
    unittest.main()
