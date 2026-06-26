"""Tests for the pytree batch API in stratum/batch.py."""

import unittest

import torch

from stratum.batch import (
    AVERAGE,
    CHUNK_DIM0,
    REPLICATE,
    TokenWeightedReducer,
    guess_split_spec,
    merge_pytree,
    split_kwargs_pytree,
    split_pytree,
)


class GuessSplitSpecTest(unittest.TestCase):
    def test_guesses_chunk_for_matching_batch_tensors(self):
        batch = {
            "input_ids": torch.zeros(4, 8),
            "attention_mask": torch.ones(4, 8),
            "labels": torch.full((4, 8), -100),
        }
        spec, bs = guess_split_spec(batch)
        self.assertEqual(bs, 4)
        self.assertIs(spec["input_ids"], CHUNK_DIM0)
        self.assertIs(spec["attention_mask"], CHUNK_DIM0)
        self.assertIs(spec["labels"], CHUNK_DIM0)

    def test_replicates_non_tensors(self):
        batch = {"input_ids": torch.zeros(4, 8), "config": {"foo": "bar"}}
        spec, bs = guess_split_spec(batch)
        # Nested dicts are recursed into; the leaf is REPLICATE.
        self.assertIs(spec["config"]["foo"], REPLICATE)

    def test_marks_scalars_as_average(self):
        batch = {"loss": torch.tensor(1.5)}
        spec, bs = guess_split_spec(batch)
        self.assertIs(spec["loss"], AVERAGE)

    def test_respects_expected_batch_size(self):
        batch = {
            "a": torch.zeros(4, 8),
            "b": torch.zeros(2, 8),  # different batch dim
        }
        spec, bs = guess_split_spec(batch, expected_batch_size=4)
        self.assertIs(spec["a"], CHUNK_DIM0)
        self.assertIs(spec["b"], REPLICATE)

    def test_nested_dict(self):
        batch = {
            "input": {"ids": torch.zeros(4, 8), "mask": torch.ones(4, 8)},
            "labels": torch.full((4, 8), -100),
        }
        spec, bs = guess_split_spec(batch)
        self.assertEqual(bs, 4)
        self.assertIs(spec["input"]["ids"], CHUNK_DIM0)
        self.assertIs(spec["input"]["mask"], CHUNK_DIM0)
        self.assertIs(spec["labels"], CHUNK_DIM0)

    def test_list_of_tensors(self):
        batch = [torch.zeros(4, 8), torch.ones(4, 8)]
        spec, bs = guess_split_spec(batch)
        self.assertEqual(bs, 4)
        for entry in spec:
            self.assertIs(entry, CHUNK_DIM0)


class SplitPytreeTest(unittest.TestCase):
    def test_splits_dict_into_two_microbatches(self):
        batch = {
            "input_ids": torch.arange(8).reshape(4, 2),
            "labels": torch.arange(4),
        }
        chunks = split_pytree(batch, num_chunks=2)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(torch.equal(chunks[0]["input_ids"], batch["input_ids"][:2]))
        self.assertTrue(torch.equal(chunks[1]["input_ids"], batch["input_ids"][2:]))
        self.assertTrue(torch.equal(chunks[0]["labels"], batch["labels"][:2]))
        self.assertTrue(torch.equal(chunks[1]["labels"], batch["labels"][2:]))

    def test_splits_into_three_with_remainder(self):
        batch = {"x": torch.arange(10).reshape(5, 2)}
        chunks = split_pytree(batch, num_chunks=3)
        self.assertEqual(len(chunks), 3)
        sizes = [c["x"].size(0) for c in chunks]
        self.assertEqual(sizes, [2, 2, 1])

    def test_replicates_non_tensor_values(self):
        batch = {"x": torch.arange(4), "cfg": {"lr": 1e-4}}
        chunks = split_pytree(batch, num_chunks=2)
        self.assertEqual(chunks[0]["cfg"]["lr"], 1e-4)
        self.assertEqual(chunks[1]["cfg"]["lr"], 1e-4)

    def test_single_chunk_returns_original(self):
        batch = {"x": torch.arange(4)}
        chunks = split_pytree(batch, num_chunks=1)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(torch.equal(chunks[0]["x"], batch["x"]))

    def test_explicit_spec_overrides_guess(self):
        batch = {"x": torch.arange(4), "y": torch.arange(4)}
        spec, _ = guess_split_spec(batch)
        # Override: replicate y
        spec["y"] = REPLICATE
        chunks = split_pytree(batch, num_chunks=2, spec=spec)
        self.assertTrue(torch.equal(chunks[0]["x"], torch.arange(2)))
        self.assertTrue(torch.equal(chunks[0]["y"], torch.arange(4)))
        self.assertTrue(torch.equal(chunks[1]["y"], torch.arange(4)))

    def test_nested_dict_split(self):
        batch = {
            "input": {"ids": torch.arange(6).reshape(3, 2)},
            "labels": torch.arange(3),
        }
        chunks = split_pytree(batch, num_chunks=3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["input"]["ids"].size(0), 1)
        self.assertEqual(chunks[0]["labels"].size(0), 1)

    def test_rejects_invalid_num_chunks(self):
        with self.assertRaises(ValueError):
            split_pytree({"x": torch.arange(4)}, num_chunks=0)


class MergePytreeTest(unittest.TestCase):
    def test_merges_chunked_tensors(self):
        chunks = [
            {"x": torch.tensor([0, 1])},
            {"x": torch.tensor([2, 3])},
        ]
        merged = merge_pytree(chunks)
        self.assertTrue(torch.equal(merged["x"], torch.tensor([0, 1, 2, 3])))

    def test_averages_scalar_tensors(self):
        chunks = [
            {"loss": torch.tensor(2.0)},
            {"loss": torch.tensor(4.0)},
        ]
        merged = merge_pytree(chunks)
        self.assertAlmostEqual(float(merged["loss"]), 3.0)

    def test_replicated_values_take_first(self):
        chunks = [
            {"x": torch.tensor([0]), "cfg": "a"},
            {"x": torch.tensor([1]), "cfg": "a"},
        ]
        merged = merge_pytree(chunks)
        self.assertEqual(merged["cfg"], "a")

    def test_single_chunk_returns_as_is(self):
        chunk = {"x": torch.tensor([0, 1])}
        merged = merge_pytree([chunk])
        self.assertTrue(torch.equal(merged["x"], chunk["x"]))

    def test_roundtrip_split_then_merge(self):
        batch = {
            "input_ids": torch.arange(12).reshape(6, 2),
            "labels": torch.arange(6),
        }
        chunks = split_pytree(batch, num_chunks=3)
        merged = merge_pytree(chunks)
        self.assertTrue(torch.equal(merged["input_ids"], batch["input_ids"]))
        self.assertTrue(torch.equal(merged["labels"], batch["labels"]))

    def test_rejects_empty_chunks(self):
        with self.assertRaises(ValueError):
            merge_pytree([])


class SplitKwargsPytreeTest(unittest.TestCase):
    def test_splits_kwargs_dict(self):
        kwargs = {
            "input_ids": torch.arange(8).reshape(4, 2),
            "attention_mask": torch.ones(4, 2),
            "labels": torch.arange(4),
            "return_logits": False,
        }
        result = split_kwargs_pytree(kwargs, num_microbatch=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["input_ids"].size(0), 2)
        self.assertEqual(result[1]["input_ids"].size(0), 2)
        self.assertFalse(result[0]["return_logits"])
        self.assertFalse(result[1]["return_logits"])


class TokenWeightedReducerTest(unittest.TestCase):
    def test_token_weighted_average(self):
        r = TokenWeightedReducer()
        r.accumulate(torch.tensor(2.0), 3)
        r.accumulate(torch.tensor(10.0), 1)
        result = r.reduce()
        self.assertAlmostEqual(float(result), 4.0)

    def test_single_accumulation(self):
        r = TokenWeightedReducer()
        r.accumulate(torch.tensor(5.0), 10)
        result = r.reduce()
        self.assertAlmostEqual(float(result), 5.0)

    def test_all_ignored_tokens_falls_back_to_simple_average(self):
        r = TokenWeightedReducer()
        r.accumulate(torch.tensor(2.0), 0)
        r.accumulate(torch.tensor(10.0), 0)
        result = r.reduce()
        self.assertAlmostEqual(float(result), 6.0)

    def test_raises_on_empty(self):
        r = TokenWeightedReducer()
        with self.assertRaises(ValueError):
            r.reduce()


if __name__ == "__main__":
    unittest.main()
