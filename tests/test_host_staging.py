import unittest

import torch

from stratum.host_staging import _typed_buffer_view


class HostStagingBufferTest(unittest.TestCase):
    def test_typed_view_uses_byte_count_for_wider_dtypes(self):
        buffer = torch.empty(64, dtype=torch.uint8)

        float32_view = _typed_buffer_view(buffer, torch.float32, 8)
        int64_view = _typed_buffer_view(buffer, torch.int64, 4)

        self.assertEqual(float32_view.dtype, torch.float32)
        self.assertEqual(float32_view.numel(), 8)
        self.assertEqual(int64_view.dtype, torch.int64)
        self.assertEqual(int64_view.numel(), 4)


if __name__ == "__main__":
    unittest.main()
