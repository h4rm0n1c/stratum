"""Tests for the sample packing collation in stratum/packing.py."""

import unittest

import torch

from stratum.batch import training_token_counts
from stratum.packing import (
    compute_cu_seqlens,
    compute_seq_idx,
    pack_collate,
    pack_samples,
    split_packed_batch,
    unpack_positions,
)


class ComputeCuSeqlensTest(unittest.TestCase):
    def test_basic_cumulative_lengths(self):
        cu = compute_cu_seqlens([3, 4, 2])
        self.assertTrue(torch.equal(cu, torch.tensor([0, 3, 7, 9], dtype=torch.int32)))

    def test_empty_lengths(self):
        cu = compute_cu_seqlens([])
        self.assertTrue(torch.equal(cu, torch.tensor([0], dtype=torch.int32)))

    def test_single_sample(self):
        cu = compute_cu_seqlens([10])
        self.assertTrue(torch.equal(cu, torch.tensor([0, 10], dtype=torch.int32)))


class UnpackPositionsTest(unittest.TestCase):
    def test_resets_at_boundaries(self):
        cu = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
        pos = unpack_positions(cu)
        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1], dtype=torch.int64)
        self.assertTrue(torch.equal(pos, expected))

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        pos = unpack_positions(cu)
        self.assertTrue(torch.equal(pos, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)))

    def test_empty(self):
        cu = torch.tensor([0], dtype=torch.int32)
        pos = unpack_positions(cu)
        self.assertEqual(pos.numel(), 0)


class PackSamplesTest(unittest.TestCase):
    def _make_sample(self, length, label_start=0):
        return {
            "input_ids": torch.arange(length),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.arange(label_start, label_start + length),
        }

    def test_packs_multiple_samples(self):
        samples = [self._make_sample(3), self._make_sample(4), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=20)

        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["input_ids"].numel(), 9)
        self.assertTrue(torch.equal(result["cu_seqlens"], torch.tensor([0, 3, 7, 9], dtype=torch.int32)))

    def test_position_ids_reset_at_boundaries(self):
        samples = [self._make_sample(3), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=20)

        expected_pos = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
        self.assertTrue(torch.equal(result["position_ids"], expected_pos))

    def test_drops_samples_exceeding_max_seq_len(self):
        samples = [self._make_sample(5), self._make_sample(4), self._make_sample(3)]
        result = pack_samples(samples, max_seq_len=10)

        # 5 + 4 = 9 fits, 9 + 3 = 12 exceeds
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["input_ids"].numel(), 9)

    def test_labels_are_packed_correctly(self):
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([40, 50])},
        ]
        # mask_boundaries=False: verify raw concatenation without the boundary guard
        result = pack_samples(samples, max_seq_len=20, mask_boundaries=False)
        expected_labels = torch.tensor([10, 20, 30, 40, 50])
        self.assertTrue(torch.equal(result["labels"], expected_labels))

    def test_boundary_label_is_masked_by_default(self):
        """Default packing masks the first label of each sample after the first."""
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([40, 50])},
        ]
        result = pack_samples(samples, max_seq_len=20)
        # Position 3 is the first label of the second sample — masked to -100.
        self.assertEqual(int(result["labels"][3].item()), -100)
        # Other labels unchanged.
        self.assertEqual(int(result["labels"][0].item()), 10)
        self.assertEqual(int(result["labels"][4].item()), 50)

    def test_max_seqlen_is_longest_sample(self):
        samples = [self._make_sample(3), self._make_sample(7), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=20)
        self.assertEqual(result["max_seqlen"], 7)

    def test_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            pack_samples([], max_seq_len=10)

    def test_rejects_when_no_samples_fit(self):
        samples = [self._make_sample(100)]
        with self.assertRaises(ValueError):
            pack_samples(samples, max_seq_len=10)

    def test_greedy_stops_at_first_overflow(self):
        """greedy strategy stops at the first sample that doesn't fit."""
        # 3 + 10 = 13 > 12, so greedy stops after the first sample
        samples = [self._make_sample(3), self._make_sample(10), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=12, strategy="greedy")
        self.assertEqual(result["n_samples"], 1)

    def test_first_fit_skips_oversized_samples(self):
        """first_fit skips samples that don't fit and packs smaller ones."""
        # 3 + 10 > 8 (overflow), but 3 + 2 = 5 fits
        samples = [self._make_sample(3), self._make_sample(10), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=8, strategy="first_fit")
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["input_ids"].numel(), 5)

    def test_boundary_masking_sets_first_label_of_each_segment_to_ignore(self):
        """Labels at packed-sample boundaries are set to ignore_index."""
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([10, 20, 30])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([40, 50])},
            {"input_ids": torch.tensor([6]), "labels": torch.tensor([60])},
        ]
        result = pack_samples(samples, max_seq_len=20, mask_boundaries=True)
        # Segment 1 starts at index 3, segment 2 starts at index 5
        self.assertEqual(int(result["labels"][3].item()), -100)
        self.assertEqual(int(result["labels"][5].item()), -100)
        # Labels inside segments unchanged
        self.assertEqual(int(result["labels"][0].item()), 10)
        self.assertEqual(int(result["labels"][4].item()), 50)

    def test_no_boundary_masking_when_disabled(self):
        """mask_boundaries=False leaves labels untouched."""
        samples = [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([10, 20])},
            {"input_ids": torch.tensor([3, 4]), "labels": torch.tensor([30, 40])},
        ]
        result = pack_samples(samples, max_seq_len=20, mask_boundaries=False)
        self.assertEqual(int(result["labels"][2].item()), 30)

    def test_boundary_masking_single_sample_unchanged(self):
        """Boundary masking has no effect when there is only one sample."""
        samples = [self._make_sample(5, label_start=10)]
        result = pack_samples(samples, max_seq_len=20, mask_boundaries=True)
        self.assertFalse(any(l == -100 for l in result["labels"].tolist()))


class PackCollateTest(unittest.TestCase):
    def _make_sample(self, length):
        return {
            "input_ids": torch.arange(length),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.arange(length),
        }

    def test_collates_batch(self):
        batch = [self._make_sample(3), self._make_sample(5), self._make_sample(2)]
        result = pack_collate(batch, max_seq_len=20)

        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["input_ids"].numel(), 10)
        self.assertEqual(result["max_seqlen"], 5)


class PackedBatchFormatTest(unittest.TestCase):
    """Verify the packed batch format matches what the pipeline expects."""

    def _make_sample(self, length, label_start=0):
        return {
            "input_ids": torch.arange(length),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.arange(label_start, label_start + length),
        }

    def test_packed_attention_mask_is_dict_with_cu_seqlens(self):
        """Packed mode passes cu_seqlens through attention_mask dict."""
        samples = [self._make_sample(3), self._make_sample(4)]
        result = pack_samples(samples, max_seq_len=20)

        attention_mask = {
            "cu_seqlens": result["cu_seqlens"],
            "max_seqlen": result["max_seqlen"],
        }

        self.assertIn("cu_seqlens", attention_mask)
        self.assertIn("max_seqlen", attention_mask)
        self.assertEqual(attention_mask["cu_seqlens"].dtype, torch.int32)

    def test_packed_position_ids_match_cu_seqlens(self):
        """Position IDs should reset at each cu_seqlens boundary."""
        samples = [self._make_sample(3), self._make_sample(5), self._make_sample(2)]
        result = pack_samples(samples, max_seq_len=20)

        cu = result["cu_seqlens"]
        pos = result["position_ids"]

        # Verify position_ids reset at each boundary
        for i in range(len(cu) - 1):
            start = cu[i].item()
            end = cu[i + 1].item()
            expected = torch.arange(end - start, dtype=torch.int64)
            self.assertTrue(torch.equal(pos[start:end], expected))

    def test_packed_input_ids_are_1d(self):
        """Packed input_ids should be 1D."""
        samples = [self._make_sample(3), self._make_sample(4)]
        result = pack_samples(samples, max_seq_len=20)
        self.assertEqual(result["input_ids"].dim(), 1)

    def test_packed_labels_are_1d(self):
        """Packed labels should be 1D."""
        samples = [self._make_sample(3), self._make_sample(4)]
        result = pack_samples(samples, max_seq_len=20)
        self.assertEqual(result["labels"].dim(), 1)

    def test_training_token_counts_accepts_packed_attention_metadata(self):
        # Sample 1 labels: [-100, 1, 2]  → 2 trainable
        # Sample 2 labels: [3, -100, 5, 6] → 3 trainable originally
        # Boundary masking sets label[3] (first of sample 2) to -100 → 2 trainable
        # Total trainable: 2 + 2 = 4
        samples = [
            {
                "input_ids": torch.arange(3),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "labels": torch.tensor([-100, 1, 2]),
            },
            {
                "input_ids": torch.arange(4),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor([3, -100, 5, 6]),
            },
        ]
        packed = pack_samples(samples, max_seq_len=20)
        attention_mask = {
            "cu_seqlens": packed["cu_seqlens"],
            "max_seqlen": packed["max_seqlen"],
        }

        total, trainable = training_token_counts(
            packed["input_ids"], attention_mask, packed["labels"]
        )

        self.assertEqual(total, 7)
        self.assertEqual(trainable, 4)  # boundary label masked to -100


class SplitPackedBatchTest(unittest.TestCase):
    """Tests for splitting packed batches into microbatches."""

    def _make_packed(self, *lengths):
        ids = [torch.arange(n) for n in lengths]
        labels = [torch.full((n,), 100 + i * 10) for i, n in enumerate(lengths)]
        return pack_samples(
            [
                {"input_ids": i, "attention_mask": torch.ones_like(i), "labels": l}
                for i, l in zip(ids, labels)
            ],
            max_seq_len=sum(lengths) * 2,
        )

    def test_splits_by_sample_boundary(self):
        """Microbatches should each contain whole samples."""
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)

        self.assertEqual(len(mbs), 2)
        # 4 samples split into 2 groups: [3, 5] and [2, 4]
        self.assertEqual(mbs[0]["n_samples"], 2)
        self.assertEqual(mbs[1]["n_samples"], 2)

    def test_cu_seqlens_renormalised(self):
        """cu_seqlens should start at 0 in each microbatch."""
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)

        for mb in mbs:
            self.assertEqual(int(mb["cu_seqlens"][0].item()), 0)

    def test_three_microbatches_with_remainder(self):
        """5 samples split into 3 groups: 2, 2, 1."""
        packed = self._make_packed(3, 2, 4, 5, 3)
        mbs = split_packed_batch(packed, num_microbatch=3)

        self.assertEqual(len(mbs), 3)
        self.assertEqual([mb["n_samples"] for mb in mbs], [2, 2, 1])

    def test_single_microbatch_returns_original(self):
        """num_microbatch=1 should return one batch with all samples."""
        packed = self._make_packed(3, 5, 2)
        mbs = split_packed_batch(packed, num_microbatch=1)

        self.assertEqual(len(mbs), 1)
        self.assertEqual(mbs[0]["n_samples"], 3)
        self.assertTrue(torch.equal(mbs[0]["input_ids"], packed["input_ids"]))

    def test_trainable_tokens_computed(self):
        """Each microbatch should have trainable_tokens."""
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)

        for mb in mbs:
            self.assertIn("trainable_tokens", mb)
            expected = int((mb["labels"] != -100).sum().item())
            self.assertEqual(mb["trainable_tokens"], expected)

    def test_total_tokens_preserved(self):
        """Sum of microbatch token counts should match original."""
        packed = self._make_packed(3, 5, 2, 4)
        total = packed["input_ids"].numel()
        mbs = split_packed_batch(packed, num_microbatch=3)

        restored = sum(mb["input_ids"].numel() for mb in mbs)
        self.assertEqual(restored, total)

    def test_roundtrip_positions(self):
        """Position IDs should be correct after split."""
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)

        for mb in mbs:
            cu = mb["cu_seqlens"]
            pos = mb["position_ids"]
            for i in range(len(cu) - 1):
                start = int(cu[i])
                end = int(cu[i + 1])
                expected = torch.arange(end - start, dtype=torch.int64)
                self.assertTrue(
                    torch.equal(pos[start:end], expected),
                    f"position_ids not reset at boundary {i}: "
                    f"got {pos[start:end]}, expected {expected}"
                )


class ComputeSeqIdxTest(unittest.TestCase):
    def test_basic(self):
        cu = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        seq_idx = compute_seq_idx(cu)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=torch.int32)
        self.assertTrue(torch.equal(seq_idx, expected))

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        seq_idx = compute_seq_idx(cu)
        self.assertTrue(torch.equal(seq_idx, torch.zeros(5, dtype=torch.int32)))


class SplitPackedBatchSeqIdxTest(unittest.TestCase):
    def _make_packed(self, *lengths):
        ids = [torch.arange(n) for n in lengths]
        labels = [torch.full((n,), 100 + i * 10) for i, n in enumerate(lengths)]
        return pack_samples(
            [{"input_ids": i, "attention_mask": torch.ones_like(i), "labels": l}
             for i, l in zip(ids, labels)],
            max_seq_len=sum(lengths) * 2,
        )

    def test_seq_idx_propagated_to_microbatches(self):
        packed = self._make_packed(3, 5, 2, 4)
        self.assertIsNotNone(packed["seq_idx"])
        mbs = split_packed_batch(packed, num_microbatch=2)
        for mb in mbs:
            self.assertIn("seq_idx", mb)
            self.assertIsNotNone(mb["seq_idx"])

    def test_seq_idx_starts_at_zero_in_each_microbatch(self):
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)
        for mb in mbs:
            self.assertEqual(int(mb["seq_idx"][0].item()), 0,
                             "seq_idx must start at 0 in each microbatch")

    def test_seq_idx_values_match_cu_seqlens(self):
        """seq_idx value at each token matches its sample index within the microbatch."""
        packed = self._make_packed(3, 5, 2, 4)
        mbs = split_packed_batch(packed, num_microbatch=2)
        for mb in mbs:
            cu = mb["cu_seqlens"]
            seq_idx = mb["seq_idx"]
            for seg_i in range(mb["n_samples"]):
                start = int(cu[seg_i])
                end = int(cu[seg_i + 1])
                for tok in range(start, end):
                    self.assertEqual(int(seq_idx[tok].item()), seg_i)

    def test_single_microbatch_seq_idx_preserved(self):
        packed = self._make_packed(3, 5, 2)
        mbs = split_packed_batch(packed, num_microbatch=1)
        self.assertTrue(torch.equal(mbs[0]["seq_idx"], packed["seq_idx"]))

    def test_single_sample_pack_seq_idx_is_none(self):
        packed = self._make_packed(5)
        self.assertIsNone(packed["seq_idx"])
        mbs = split_packed_batch(packed, num_microbatch=1)
        self.assertIsNone(mbs[0]["seq_idx"])


if __name__ == "__main__":
    unittest.main()
