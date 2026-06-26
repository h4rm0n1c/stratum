import unittest

import torch

from stratum.transfer import (
    PinnedUpload,
    RegisterBackwardEvent,
    TransferResult,
    async_d2h,
    async_h2d,
)


class FakeEvent:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class TransferTests(unittest.TestCase):
    def test_transfer_result_wait_synchronizes_event(self):
        event = FakeEvent()
        tensor = torch.tensor([1.0])
        result = TransferResult(tensor, event=event)

        self.assertIs(result.wait(), tensor)
        self.assertTrue(event.synchronized)

    def test_async_h2d_cpu_path_copies_without_mutating_requires_grad(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = async_h2d(tensor, "cpu", keep_requires_grad=False)

        self.assertEqual(result.tensor.device.type, "cpu")
        self.assertFalse(result.tensor.requires_grad)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(torch.equal(result.tensor, tensor.detach()))
        self.assertIsNone(result.event)

    def test_async_h2d_cpu_path_can_keep_requires_grad(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = async_h2d(tensor, "cpu", keep_requires_grad=True)

        self.assertTrue(result.tensor.requires_grad)
        self.assertIsNot(result.tensor, tensor)

    def test_async_h2d_cpu_path_can_preserve_autograd(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = async_h2d(tensor, "cpu", preserve_autograd=True)

        result.tensor.sum().backward()

        self.assertTrue(result.tensor.requires_grad)
        self.assertIsNotNone(result.tensor.grad_fn)
        self.assertTrue(torch.equal(tensor.grad, torch.tensor([1.0, 1.0])))

    def test_async_h2d_cuda_unavailable_is_explicit(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")

        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            async_h2d(torch.tensor([1.0]), "cuda:0")

    def test_async_d2h_cpu_path_copies_without_mutating_requires_grad(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = async_d2h(tensor, keep_requires_grad=False)

        self.assertEqual(result.tensor.device.type, "cpu")
        self.assertFalse(result.tensor.requires_grad)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(torch.equal(result.tensor, tensor.detach()))
        self.assertIsNone(result.event)

    def test_async_d2h_cpu_path_can_preserve_autograd(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = async_d2h(tensor, preserve_autograd=True)

        (result.tensor * 2.0).sum().backward()

        self.assertTrue(result.tensor.requires_grad)
        self.assertIsNotNone(result.tensor.grad_fn)
        self.assertTrue(torch.equal(tensor.grad, torch.tensor([2.0, 2.0])))

    def test_pinned_upload_cpu_autograd(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        uploaded = PinnedUpload.apply(tensor, torch.device("cpu"))

        (uploaded * 3.0).sum().backward()

        self.assertTrue(torch.equal(tensor.grad, torch.tensor([3.0, 3.0])))

    def test_register_backward_event_waits(self):
        event = FakeEvent()
        tensor = torch.tensor([1.0], requires_grad=True)
        guarded = RegisterBackwardEvent.apply(tensor, event)

        (guarded * 2.0).sum().backward()

        self.assertTrue(event.synchronized)
        self.assertTrue(torch.equal(tensor.grad, torch.tensor([2.0])))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_async_h2d_cuda_roundtrip(self):
        tensor = torch.tensor([1.0, 2.0])
        result = async_h2d(tensor, "cuda:0")

        device_tensor = result.wait()

        self.assertEqual(device_tensor.device.type, "cuda")
        self.assertTrue(torch.equal(device_tensor.cpu(), tensor))
        self.assertIsNotNone(result.event)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_async_d2h_cuda_roundtrip(self):
        tensor = torch.tensor([1.0, 2.0], device="cuda:0")
        result = async_d2h(tensor)

        host_tensor = result.wait()

        self.assertEqual(host_tensor.device.type, "cpu")
        self.assertTrue(torch.equal(host_tensor, tensor.cpu()))
        self.assertTrue(host_tensor.is_pinned())
        self.assertIsNotNone(result.event)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_async_cuda_preserve_autograd_with_preallocated_buffers(self):
        source = torch.tensor([1.0, 2.0], device="cuda:0", requires_grad=True)
        stream = torch.cuda.current_stream(device=source.device)
        host_out = torch.empty_like(source, device="cpu", pin_memory=True)
        d2h = async_d2h(
            source,
            stream=stream,
            preserve_autograd=True,
            out=host_out,
        )
        host_tensor = d2h.wait()

        device_out = torch.empty_like(source, device="cuda:0")
        h2d = async_h2d(
            host_tensor,
            "cuda:0",
            stream=stream,
            preserve_autograd=True,
            out=device_out,
        )
        device_tensor = h2d.wait()
        (device_tensor * 3.0).sum().backward()

        self.assertIs(host_tensor, host_out)
        self.assertIs(device_tensor, device_out)
        self.assertTrue(torch.equal(source.grad, torch.tensor([3.0, 3.0], device="cuda:0")))


if __name__ == "__main__":
    unittest.main()
