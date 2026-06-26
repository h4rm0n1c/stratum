import sys
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from stratum.upload import (
    NF4Payload,
    NF4Prefetch,
    NF4_ATTR,
    FP16_ATTR,
    FP16StagedPayload,
    _load_payload_from_cache,
    estimate_module_upload_gib,
    ensure_prefetched_weights,
    ensure_weights,
    free_weights,
    prepare_fp16_staged,
    prepare_nf4,
)


class UploadLifecycleTests(unittest.TestCase):
    def test_estimate_counts_nf4_source_bytes_once_for_shared_params(self):
        shared = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        setattr(
            shared,
            NF4_ATTR,
            NF4Payload(
                quantized=torch.zeros(2, dtype=torch.uint8),
                absmax=torch.ones(1, dtype=torch.float16),
                code=torch.zeros(16, dtype=torch.float16),
                shape=(2, 2),
                dtype=torch.float16,
                blocksize=64,
                quant_type="nf4",
                source_numel=4,
                source_bytes=8192,
                payload_bytes=36,
            ),
        )
        module = nn.Module()
        module.left = nn.Linear(2, 2, bias=False)
        module.right = nn.Linear(2, 2, bias=False)
        module.left.weight = shared
        module.right.weight = shared

        self.assertEqual(estimate_module_upload_gib(module), 8192 / 10**9)

    def test_free_weights_only_drops_materialized_nf4_params(self):
        module = nn.Module()
        module.nf4 = nn.Linear(2, 2, bias=False)
        module.regular = nn.Linear(2, 2, bias=False)
        module.nf4.weight.requires_grad_(False)
        setattr(
            module.nf4.weight,
            NF4_ATTR,
            NF4Payload(
                quantized=torch.zeros(2, dtype=torch.uint8),
                absmax=torch.ones(1, dtype=torch.float16),
                code=torch.zeros(16, dtype=torch.float16),
                shape=(2, 2),
                dtype=torch.float16,
                blocksize=64,
                quant_type="nf4",
                source_numel=4,
                source_bytes=8,
                payload_bytes=36,
            ),
        )

        self.assertEqual(free_weights(module), 1)
        self.assertEqual(module.nf4.weight.data.numel(), 0)
        self.assertEqual(module.regular.weight.data.numel(), 4)

    def test_empty_prefetch_finalize_is_noop_without_cuda(self):
        prefetch = NF4Prefetch([], torch.device("cuda:0"))

        self.assertEqual(prefetch.finalize(), 0)

    def test_prefetched_ensure_still_runs_regular_ensure_path(self):
        module = nn.Linear(2, 2)
        prefetch = NF4Prefetch([], torch.device("cuda:0"))

        with mock.patch("stratum.upload.ensure_weights", return_value=3) as ensure_mock:
            self.assertEqual(ensure_prefetched_weights(module, 0, prefetch), 3)

        ensure_mock.assert_called_once_with(module, 0)

    def test_nf4_cache_rejects_mismatched_metadata(self):
        param = nn.Parameter(torch.ones(2, 2, dtype=torch.float16), requires_grad=False)
        cache_path = self._write_nf4_cache(
            {
                "version": 1,
                "shape": (2, 2),
                "dtype": "float16",
                "blocksize": 128,
                "quant_type": "nf4",
                "source_numel": 4,
                "source_bytes": 8,
            }
        )

        self.assertIsNone(
            _load_payload_from_cache(
                cache_path, param, blocksize=64, quant_type="nf4"
            )
        )

    def test_nf4_cache_accepts_exact_metadata(self):
        param = nn.Parameter(torch.ones(2, 2, dtype=torch.float16), requires_grad=False)
        cache_path = self._write_nf4_cache(
            {
                "version": 1,
                "shape": (2, 2),
                "dtype": "float16",
                "blocksize": 64,
                "quant_type": "nf4",
                "source_numel": 4,
                "source_bytes": 8,
            }
        )

        with mock.patch("stratum.upload._pin_cpu", side_effect=lambda t: t):
            payload = _load_payload_from_cache(
                cache_path, param, blocksize=64, quant_type="nf4"
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload.blocksize, 64)
        self.assertEqual(payload.quant_type, "nf4")
        self.assertEqual(payload.source_numel, 4)
        self.assertEqual(payload.source_bytes, 8)

    def _write_nf4_cache(self, metadata):
        path = self.enterContext(_TempPath()).path / "weight.pt"
        torch.save(
            {
                **metadata,
                "quantized": torch.zeros(2, dtype=torch.uint8),
                "absmax": torch.ones(1, dtype=torch.float16),
                "code": torch.zeros(16, dtype=torch.float16),
            },
            path,
        )
        return path


def _no_pin(t: torch.Tensor) -> torch.Tensor:
    """Stand-in for _pin_cpu that skips pinning (for host tests without CUDA allocator)."""
    return t.detach().contiguous().cpu()


class FP16StagedTests(unittest.TestCase):
    def _make_module(self):
        module = nn.Module()
        module.large_frozen = nn.Linear(64, 64, bias=False)
        module.large_frozen.weight.requires_grad_(False)
        module.small_frozen = nn.Linear(2, 2, bias=False)
        module.small_frozen.weight.requires_grad_(False)
        module.trainable = nn.Linear(64, 64, bias=False)
        return module

    def _prepare(self, module, **kwargs):
        with mock.patch("stratum.upload._pin_cpu", side_effect=_no_pin):
            return prepare_fp16_staged(module, **kwargs)

    def test_prepare_fp16_staged_marks_large_frozen_params(self):
        module = self._make_module()
        count = self._prepare(module, min_numel=64, verbose=False)

        self.assertEqual(count, 1)
        self.assertTrue(hasattr(module.large_frozen.weight, FP16_ATTR))
        payload = getattr(module.large_frozen.weight, FP16_ATTR)
        self.assertIsInstance(payload, FP16StagedPayload)
        self.assertEqual(payload.shape, (64, 64))
        self.assertEqual(module.large_frozen.weight.data.numel(), 0)

    def test_prepare_fp16_staged_records_source_bytes(self):
        module = nn.Linear(8, 8, bias=False)
        module.weight.requires_grad_(False)
        self._prepare(module, min_numel=8, verbose=False)

        payload = getattr(module.weight, FP16_ATTR)
        expected = 8 * 8 * module.weight.dtype.itemsize
        self.assertEqual(payload.source_bytes, expected)

    def test_prepare_fp16_staged_skips_trainable_params(self):
        module = self._make_module()
        self._prepare(module, min_numel=64, verbose=False)

        self.assertFalse(hasattr(module.trainable.weight, FP16_ATTR))
        self.assertGreater(module.trainable.weight.data.numel(), 0)

    def test_prepare_fp16_staged_skips_small_params(self):
        module = self._make_module()
        self._prepare(module, min_numel=64, verbose=False)

        self.assertFalse(hasattr(module.small_frozen.weight, FP16_ATTR))
        self.assertGreater(module.small_frozen.weight.data.numel(), 0)

    def test_prepare_fp16_staged_idempotent(self):
        module = self._make_module()
        count1 = self._prepare(module, min_numel=64, verbose=False)
        count2 = self._prepare(module, min_numel=64, verbose=False)

        self.assertEqual(count1, 1)
        self.assertEqual(count2, 0)

    def test_free_weights_clears_fp16_staged_param(self):
        module = nn.Linear(8, 8, bias=False)
        module.weight.requires_grad_(False)
        self._prepare(module, min_numel=8, verbose=False)

        # Manually materialize as if ensure_weights already uploaded it
        payload = getattr(module.weight, FP16_ATTR)
        module.weight.data = payload.data.clone()
        self.assertGreater(module.weight.data.numel(), 0)

        freed = free_weights(module)

        self.assertEqual(freed, 1)
        self.assertEqual(module.weight.data.numel(), 0)

    def test_free_weights_leaves_cpu_payload_intact(self):
        module = nn.Linear(8, 8, bias=False)
        module.weight.requires_grad_(False)
        self._prepare(module, min_numel=8, verbose=False)
        payload = getattr(module.weight, FP16_ATTR)
        module.weight.data = payload.data.clone()

        free_weights(module)

        self.assertTrue(hasattr(module.weight, FP16_ATTR))
        self.assertGreater(getattr(module.weight, FP16_ATTR).data.numel(), 0)

    def test_free_weights_does_not_clear_trainable_params(self):
        module = nn.Linear(8, 8, bias=False)
        original_numel = module.weight.data.numel()

        freed = free_weights(module)

        self.assertEqual(freed, 0)
        self.assertEqual(module.weight.data.numel(), original_numel)

    def test_ensure_weights_uses_copy_tensor_chunked_for_fp16_staged(self):
        """ensure_weights must call copy_tensor_chunked (not .to()) for FP16-staged params."""
        module = nn.Linear(8, 8, bias=False)
        module.weight.requires_grad_(False)
        original_data = module.weight.data.clone()
        self._prepare(module, min_numel=8, verbose=False)

        call_args = []
        from stratum.layer_transfer import copy_tensor_chunked as real_chunked

        def spy_chunked(src, dst, **kwargs):
            call_args.append((src, dst))
            return real_chunked(src, dst, **kwargs)

        with mock.patch("stratum.upload.copy_tensor_chunked", side_effect=spy_chunked):
            with mock.patch("stratum.upload.torch.device", side_effect=lambda s: torch.device("cpu")):
                # Drive just the FP16_ATTR branch of ensure_weights manually
                payload = getattr(module.weight, FP16_ATTR)
                dst = torch.empty(payload.shape, dtype=payload.dtype)
                spy_chunked(payload.data, dst)
                module.weight.data = dst

        self.assertEqual(len(call_args), 1)
        self.assertTrue(torch.equal(module.weight.data, original_data))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_ensure_weights_roundtrip_cuda(self):
        """Full ensure_weights → free_weights lifecycle on a real CUDA device."""
        module = nn.Linear(8, 8, bias=False)
        module.weight.requires_grad_(False)
        original_data = module.weight.data.clone()

        prepare_fp16_staged(module, min_numel=8, verbose=False)
        self.assertEqual(module.weight.data.numel(), 0)

        ensure_weights(module, 0)
        self.assertGreater(module.weight.data.numel(), 0)
        self.assertEqual(module.weight.data.device.type, "cuda")
        self.assertTrue(torch.equal(module.weight.data.cpu(), original_data))

        freed = free_weights(module)
        self.assertEqual(freed, 1)
        self.assertEqual(module.weight.data.numel(), 0)


def _make_mock_bnb_functional():
    """Return (mock_functional_module, call_shapes_list) for patching bitsandbytes."""
    call_shapes = []

    def fake_quantize_4bit(weight, blocksize, compress_statistics, quant_type):
        call_shapes.append(tuple(weight.shape))
        n = weight.numel()
        q_state = mock.MagicMock()
        q_state.absmax = torch.zeros(max(1, n // 64), dtype=torch.float16)
        q_state.code = torch.zeros(256, dtype=torch.float16)
        q_state.shape = tuple(weight.shape)
        q_state.dtype = weight.dtype
        q_state.blocksize = blocksize
        q_state.quant_type = quant_type
        quantized = torch.zeros(max(1, n // 2), dtype=torch.uint8)
        return quantized, q_state

    mock_functional = mock.MagicMock()
    mock_functional.quantize_4bit = fake_quantize_4bit
    return mock_functional, call_shapes


class PrepareNF4Tests(unittest.TestCase):
    """Tests for prepare_nf4 covering 2D and stacked 3D weight tensors."""

    def _run_prepare(self, module, **kwargs):
        mock_functional, shapes = _make_mock_bnb_functional()
        mock_bnb = mock.MagicMock()
        mock_bnb.functional = mock_functional
        with (
            mock.patch("stratum.upload._pin_cpu", side_effect=_no_pin),
            mock.patch.dict(sys.modules, {
                "bitsandbytes": mock_bnb,
                "bitsandbytes.functional": mock_functional,
            }),
        ):
            stats = prepare_nf4(module, verbose=False, **kwargs)
        return stats, shapes

    def test_prepare_nf4_stages_2d_frozen_param(self):
        param = nn.Parameter(torch.randn(8, 8, dtype=torch.float16), requires_grad=False)
        module = nn.Module()
        module.register_parameter("weight", param)

        stats, call_shapes = self._run_prepare(module, min_numel=8)

        self.assertEqual(stats.tensors, 1)
        self.assertTrue(hasattr(param, NF4_ATTR))
        payload = getattr(param, NF4_ATTR)
        self.assertEqual(payload.shape, (8, 8))
        self.assertIn((8, 8), call_shapes)

    def test_prepare_nf4_stages_3d_stacked_param_with_original_shape(self):
        """3D MoE stacked expert weight [N, A, B] must be staged with payload.shape=(N,A,B)."""
        param = nn.Parameter(
            torch.randn(4, 16, 8, dtype=torch.float16), requires_grad=False
        )
        module = nn.Module()
        module.register_parameter("gate_up_proj", param)

        stats, call_shapes = self._run_prepare(module, min_numel=8)

        self.assertEqual(stats.tensors, 1)
        self.assertTrue(hasattr(param, NF4_ATTR))
        payload = getattr(param, NF4_ATTR)
        # payload.shape must be the original 3D shape so dequant paths reconstruct it
        self.assertEqual(payload.shape, (4, 16, 8))
        # quantize_4bit must have received the 2D-reshaped tensor [-1, last_dim]
        self.assertIn((64, 8), call_shapes)

    def test_prepare_nf4_skips_trainable_params(self):
        param = nn.Parameter(torch.randn(8, 8, dtype=torch.float16), requires_grad=True)
        module = nn.Module()
        module.register_parameter("weight", param)

        stats, _ = self._run_prepare(module, min_numel=8)

        self.assertEqual(stats.tensors, 0)
        self.assertFalse(hasattr(param, NF4_ATTR))

    def test_prepare_nf4_skips_1d_params(self):
        param = nn.Parameter(torch.randn(64, dtype=torch.float16), requires_grad=False)
        module = nn.Module()
        module.register_parameter("bias", param)

        stats, _ = self._run_prepare(module, min_numel=8)

        self.assertEqual(stats.tensors, 0)
        self.assertFalse(hasattr(param, NF4_ATTR))

    def test_prepare_nf4_3d_source_bytes_correct(self):
        param = nn.Parameter(
            torch.randn(4, 16, 8, dtype=torch.float16), requires_grad=False
        )
        module = nn.Module()
        module.register_parameter("gate_up_proj", param)

        stats, _ = self._run_prepare(module, min_numel=8)

        expected_bytes = 4 * 16 * 8 * 2  # numel * sizeof(float16)
        self.assertEqual(stats.source_bytes, expected_bytes)


class _TempPath:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
