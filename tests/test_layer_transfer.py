import unittest

import torch

from stratum.layer_transfer import (
    copy_tensor_chunked,
    download_layer_state,
    upload_layer_copies,
)


class SharedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.left = torch.nn.Linear(2, 2, bias=False)
        self.right = torch.nn.Linear(2, 2, bias=False)
        self.left.weight = shared
        self.right.weight = shared
        self.register_buffer("scale", torch.tensor([1.0, 2.0]))

    def forward(self, x):
        return self.left(x) + self.right(x * self.scale)


class LayerTransferTests(unittest.TestCase):
    def test_copy_tensor_chunked_splits_by_byte_limit(self):
        src = torch.arange(10, dtype=torch.float32)
        dst = torch.empty_like(src)

        chunks = copy_tensor_chunked(src, dst, chunk_bytes=src.element_size() * 3)

        self.assertEqual(chunks, 4)
        self.assertTrue(torch.equal(dst, src))

    def test_copy_tensor_chunked_rejects_mismatched_shape(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            copy_tensor_chunked(torch.empty(2), torch.empty(3))

    def test_upload_layer_copies_preserves_shared_parameters_and_buffers(self):
        layer = SharedLinear()

        result = upload_layer_copies([layer], "cpu", chunk_bytes=8)
        copied = result.wait()[0]

        self.assertIsNot(copied, layer)
        self.assertIs(copied.left.weight, copied.right.weight)
        self.assertIsNot(copied.left.weight, layer.left.weight)
        self.assertTrue(torch.equal(copied.left.weight, layer.left.weight))
        self.assertTrue(torch.equal(copied.scale, layer.scale))
        self.assertIsNone(result.event)

    def test_upload_layer_copies_optionally_copies_gradients(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        layer.weight.grad = torch.tensor([[3.0, 4.0]])

        copied = upload_layer_copies([layer], "cpu", with_grad=True).wait()[0]

        self.assertIsNotNone(copied.weight.grad)
        self.assertTrue(torch.equal(copied.weight.grad, layer.weight.grad))

    def test_download_layer_state_copies_gradients_and_buffers(self):
        cpu_layer = SharedLinear()
        device_layer = upload_layer_copies([cpu_layer], "cpu").wait()[0]
        device_layer.left.weight.grad = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        device_layer.scale.data = torch.tensor([9.0, 10.0])

        result = download_layer_state(cpu_layer, device_layer, with_buffer=True, with_grad=True)

        self.assertEqual(result.gradients, 1)
        self.assertEqual(result.buffers, 1)
        self.assertTrue(torch.equal(cpu_layer.left.weight.grad, device_layer.left.weight.grad))
        self.assertTrue(torch.equal(cpu_layer.scale, device_layer.scale))
        self.assertIsNone(result.event)

    def test_upload_layer_copies_cuda_unavailable_is_explicit(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")

        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            upload_layer_copies([torch.nn.Linear(1, 1)], "cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_upload_and_download_cuda_roundtrip(self):
        cpu_layer = torch.nn.Linear(2, 1, bias=False)
        copied = upload_layer_copies([cpu_layer], "cuda:0", with_grad=False).wait()[0]

        self.assertEqual(copied.weight.device.type, "cuda")
        copied.weight.grad = torch.ones_like(copied.weight)
        result = download_layer_state(cpu_layer, copied, with_grad=True).wait()

        self.assertEqual(result.gradients, 1)
        self.assertTrue(torch.equal(cpu_layer.weight.grad, torch.ones_like(cpu_layer.weight)))


if __name__ == "__main__":
    unittest.main()
