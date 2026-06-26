import unittest

import torch
import torch.nn as nn

from stratum.model.blocked_loss import BlockedPostfixCausalLMLoss


class BlockedPostfixCausalLMLossTests(unittest.TestCase):
    def test_matches_regular_shifted_lm_loss_and_hidden_grad(self):
        torch.manual_seed(2026)
        batch, seq, hidden, vocab = 1, 7, 5, 11
        norm = nn.LayerNorm(hidden)
        lm_head = nn.Linear(hidden, vocab, bias=False)
        for param in norm.parameters():
            param.requires_grad_(False)
        for param in lm_head.parameters():
            param.requires_grad_(False)

        labels = torch.randint(0, vocab, (batch, seq))
        labels[0, 3] = -100
        base_hidden = torch.randn(batch, seq, hidden)

        ref_hidden = base_hidden.clone().requires_grad_(True)
        ref_normed = norm(ref_hidden)
        ref_logits = lm_head(ref_normed[..., :-1, :])
        ref_labels = labels[..., 1:]
        ref_loss = nn.functional.cross_entropy(
            ref_logits.reshape(-1, vocab),
            ref_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / (ref_labels != -100).sum()
        ref_loss.backward()

        test_hidden = base_hidden.clone().requires_grad_(True)
        test_loss = BlockedPostfixCausalLMLoss.apply(
            test_hidden,
            labels,
            norm,
            lm_head,
            vocab,
            3,
            -100,
            False,
            False,
        )
        test_loss.backward()

        self.assertTrue(torch.allclose(test_loss, ref_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(test_hidden.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))

    def test_all_ignored_labels_returns_zero_hidden_grad(self):
        hidden = torch.randn(1, 4, 3, requires_grad=True)
        labels = torch.full((1, 4), -100)
        norm = nn.LayerNorm(3)
        lm_head = nn.Linear(3, 5, bias=False)
        for param in norm.parameters():
            param.requires_grad_(False)
        for param in lm_head.parameters():
            param.requires_grad_(False)

        loss = BlockedPostfixCausalLMLoss.apply(
            hidden,
            labels,
            norm,
            lm_head,
            5,
            2,
            -100,
            False,
            False,
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(hidden.grad, torch.zeros_like(hidden)))

    def test_none_norm_matches_identity_postfix_loss(self):
        torch.manual_seed(2026)
        batch, seq, hidden, vocab = 1, 5, 4, 9
        lm_head = nn.Linear(hidden, vocab, bias=False)
        for param in lm_head.parameters():
            param.requires_grad_(False)
        labels = torch.randint(0, vocab, (batch, seq))
        base_hidden = torch.randn(batch, seq, hidden)

        ref_hidden = base_hidden.clone().requires_grad_(True)
        ref_logits = lm_head(ref_hidden[..., :-1, :])
        ref_labels = labels[..., 1:]
        ref_loss = nn.functional.cross_entropy(
            ref_logits.reshape(-1, vocab),
            ref_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / (ref_labels != -100).sum()
        ref_loss.backward()

        test_hidden = base_hidden.clone().requires_grad_(True)
        test_loss = BlockedPostfixCausalLMLoss.apply(
            test_hidden,
            labels,
            None,
            lm_head,
            vocab,
            2,
            -100,
            False,
            False,
        )
        test_loss.backward()

        self.assertTrue(torch.allclose(test_loss, ref_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(test_hidden.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
