import unittest

import torch

from stratum.batch import (
    microbatch_loss_scale,
    reduce_microbatch_losses,
    split_training_batch,
)


class BatchHelpersTest(unittest.TestCase):
    def test_split_training_batch_balances_exact_requested_count(self):
        input_ids = torch.arange(10).reshape(5, 2)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        microbatches = split_training_batch(
            input_ids,
            attention_mask,
            labels,
            num_microbatch=2,
        )

        self.assertEqual([mb.input_ids.shape[0] for mb in microbatches], [3, 2])
        self.assertTrue(torch.equal(microbatches[0].input_ids, input_ids[:3]))
        self.assertTrue(torch.equal(microbatches[1].input_ids, input_ids[3:]))

    def test_split_training_batch_counts_trainable_tokens(self):
        input_ids = torch.arange(12).reshape(3, 4)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[0, :] = -100
        labels[2, :2] = -100

        microbatches = split_training_batch(
            input_ids,
            attention_mask,
            labels,
            num_microbatch=3,
        )

        self.assertEqual([mb.trainable_tokens for mb in microbatches], [0, 4, 2])

    def test_microbatch_loss_scale_is_token_weighted(self):
        self.assertEqual(microbatch_loss_scale(2, 10, 3), 0.2)
        self.assertEqual(microbatch_loss_scale(8, 10, 3), 0.8)

    def test_microbatch_loss_scale_falls_back_when_all_labels_ignored(self):
        self.assertAlmostEqual(microbatch_loss_scale(0, 0, 4), 0.25)

    def test_reduce_microbatch_losses_is_token_weighted(self):
        loss = reduce_microbatch_losses(
            [torch.tensor(2.0), torch.tensor(10.0)],
            [3, 1],
        )

        self.assertEqual(float(loss), 4.0)

    def test_reduce_microbatch_losses_averages_when_all_labels_ignored(self):
        loss = reduce_microbatch_losses(
            [torch.tensor(2.0), torch.tensor(10.0)],
            [0, 0],
        )

        self.assertEqual(float(loss), 6.0)

    def test_reduce_microbatch_losses_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            reduce_microbatch_losses([torch.tensor(1.0)], [1, 2])

    def test_rejects_mismatched_batch_dimensions(self):
        with self.assertRaises(ValueError):
            split_training_batch(
                torch.ones(2, 3),
                torch.ones(1, 3),
                torch.ones(2, 3),
                num_microbatch=2,
            )


if __name__ == "__main__":
    unittest.main()
