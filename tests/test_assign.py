import unittest

from stratum.assign import assign_layers_to_devices


class AssignLayersToDevicesTest(unittest.TestCase):
    def test_single_device_assignment(self):
        self.assertEqual(
            assign_layers_to_devices(4, n_devices=1),
            {0: 0, 1: 0, 2: 0, 3: 0},
        )

    def test_tensor_split_distribution_uses_requested_device_ids(self):
        assignment = assign_layers_to_devices(
            10,
            tensor_split=[1, 3],
            device_ids=[2, 7],
        )

        self.assertEqual(set(assignment), set(range(10)))
        self.assertEqual(list(assignment.values()).count(2), 3)
        self.assertEqual(list(assignment.values()).count(7), 7)

    def test_rejects_invalid_device_id_count(self):
        with self.assertRaises(ValueError):
            assign_layers_to_devices(4, tensor_split=[1, 1], device_ids=[0])


if __name__ == "__main__":
    unittest.main()
