import unittest

import torch
import torch.nn as nn

from stratum.planner import estimate_module_bytes, split_layers_by_memory_limit


class StagePlannerTest(unittest.TestCase):
    def test_estimates_module_parameter_and_buffer_bytes(self):
        module = nn.Linear(3, 4, bias=False)
        module.register_buffer("scale", torch.ones(2, dtype=torch.float32))

        expected = module.weight.numel() * module.weight.element_size()
        expected += module.scale.numel() * module.scale.element_size()

        self.assertEqual(estimate_module_bytes(module), expected)

    def test_estimate_applies_layer_size_floor(self):
        module = nn.Linear(2, 2, bias=False)
        floor_gib = 32 / 1024**3

        self.assertEqual(estimate_module_bytes(module, floor_gib=floor_gib), 32)

    def test_limit_zero_keeps_single_group(self):
        layers = [nn.Linear(2, 2), nn.Linear(2, 2)]

        self.assertEqual(split_layers_by_memory_limit(layers, 0.0), [layers])

    def test_splits_ordered_layers_under_limit(self):
        layers = [nn.Linear(8, 8, bias=False) for _ in range(3)]
        layer_gib = estimate_module_bytes(layers[0]) / 1024**3

        groups = split_layers_by_memory_limit(layers, layer_gib * 2.1)

        self.assertEqual([len(group) for group in groups], [2, 1])
        self.assertIs(groups[0][0], layers[0])
        self.assertIs(groups[0][1], layers[1])
        self.assertIs(groups[1][0], layers[2])

    def test_oversized_single_layer_gets_own_group(self):
        layers = [nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)]
        layer_gib = estimate_module_bytes(layers[0]) / 1024**3

        groups = split_layers_by_memory_limit(layers, layer_gib * 0.5)

        self.assertEqual([len(group) for group in groups], [1, 1])

    def test_layer_size_floor_affects_stage_splitting(self):
        layers = [nn.Linear(1, 1, bias=False) for _ in range(3)]
        floor_gib = 10 / 1024**3
        limit_gib = 21 / 1024**3

        groups = split_layers_by_memory_limit(
            layers,
            limit_gib,
            layer_size_floor_gib=floor_gib,
        )

        self.assertEqual([len(group) for group in groups], [2, 1])


if __name__ == "__main__":
    unittest.main()
