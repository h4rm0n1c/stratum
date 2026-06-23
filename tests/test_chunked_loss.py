import unittest

import torch
import torch.nn as nn

from stratum.model.chunked_loss import chunked_linear_cross_entropy


class ChunkedLinearCrossEntropyTest(unittest.TestCase):
    def test_matches_regular_linear_cross_entropy_loss_and_grads(self):
        torch.manual_seed(7)
        batch, seq, hidden, vocab = 2, 5, 4, 9
        base_hidden = torch.randn(batch, seq, hidden)
        base_weight = torch.randn(vocab, hidden)
        base_bias = torch.randn(vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[0, 1] = -100

        ref_hidden = base_hidden.clone().requires_grad_(True)
        ref_lm_head = nn.Linear(hidden, vocab)
        ref_lm_head.weight.data.copy_(base_weight)
        ref_lm_head.bias.data.copy_(base_bias)

        logits = ref_lm_head(ref_hidden)
        ref_loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / (labels != -100).sum()
        ref_loss.backward()

        test_hidden = base_hidden.clone().requires_grad_(True)
        test_lm_head = nn.Linear(hidden, vocab)
        test_lm_head.weight.data.copy_(base_weight)
        test_lm_head.bias.data.copy_(base_bias)

        test_loss = chunked_linear_cross_entropy(
            test_hidden,
            test_lm_head,
            labels,
            token_chunk_size=3,
        )
        test_loss.backward()

        self.assertTrue(torch.allclose(test_loss, ref_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(test_hidden.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(test_lm_head.weight.grad, ref_lm_head.weight.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(test_lm_head.bias.grad, ref_lm_head.bias.grad, atol=1e-6, rtol=1e-6))

    def test_matches_regular_linear_cross_entropy_without_bias(self):
        torch.manual_seed(11)
        hidden_states = torch.randn(1, 7, 3, requires_grad=True)
        lm_head = nn.Linear(3, 6, bias=False)
        labels = torch.randint(0, 6, (1, 7))

        ref_hidden = hidden_states.detach().clone().requires_grad_(True)
        ref_weight = lm_head.weight.detach().clone().requires_grad_(True)
        ref_logits = nn.functional.linear(ref_hidden, ref_weight)
        ref_loss = nn.functional.cross_entropy(
            ref_logits.reshape(-1, 6),
            labels.reshape(-1),
            reduction="sum",
        ) / labels.numel()
        ref_loss.backward()

        test_loss = chunked_linear_cross_entropy(
            hidden_states,
            lm_head,
            labels,
            token_chunk_size=2,
        )
        test_loss.backward()

        self.assertTrue(torch.allclose(test_loss, ref_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(hidden_states.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(lm_head.weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-6))

    def test_all_ignored_labels_returns_zero_loss(self):
        hidden_states = torch.randn(2, 3, 4, requires_grad=True)
        lm_head = nn.Linear(4, 5)
        labels = torch.full((2, 3), -100)

        loss = chunked_linear_cross_entropy(hidden_states, lm_head, labels)
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(hidden_states.grad, torch.zeros_like(hidden_states)))
        self.assertIsNone(lm_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
