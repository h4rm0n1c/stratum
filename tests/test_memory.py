import unittest

import torch
from torch import nn

from stratum.memory import pin_module_alloc, pin_module_register


class PinMemoryTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pin_helpers_skip_cuda_tensors(self):
        module = nn.Linear(2, 2).cuda()

        pin_module_alloc(module)
        pin_module_register(module)

        self.assertEqual(module.weight.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
