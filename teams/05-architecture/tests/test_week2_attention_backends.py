"""Attention execution contracts; cross-backend equality is tolerance based."""
import copy
import unittest
import torch
from test_week2_model import fixture
from week2.model import GQA, LanguageModel


class AttentionBackendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.cfg, self.rows, self.lengths = fixture()
        self.m = self.cfg["model"]

    def test_sdpa_output_and_gradients(self):
        for kv_heads in (1, 2, 4):
            for rotary in (0.5, 1.0):
                with self.subTest(kv_heads=kv_heads, rotary=rotary):
                    m = copy.deepcopy(self.m)
                    m.update(n_kv_head=kv_heads, rotary_fraction=rotary)
                    eager = GQA(m)
                    sdpa = GQA(m, global_attention="sdpa")
                    sdpa.load_state_dict(eager.state_dict())
                    x = torch.randn(2, 8, 16, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    a, b = eager(x), sdpa(y)
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                    a.square().sum().backward()
                    b.square().sum().backward()
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
                    for pa, pb in zip(eager.parameters(), sdpa.parameters()):
                        torch.testing.assert_close(pa.grad, pb.grad, atol=3e-6, rtol=3e-5)

    def test_causal_future_does_not_change_prefix(self):
        layer = GQA(self.m, global_attention="sdpa")
        x = torch.randn(2, 8, 16)
        y = x.clone()
        y[:, 4:] += 10
        torch.testing.assert_close(layer(x)[:, :4], layer(y)[:, :4], atol=0, rtol=0)

    def test_local_sink_is_exact_eager_even_when_flash_requested(self):
        m = copy.deepcopy(self.m)
        m.update(attn_pattern=["L", "G"], attn_window=3, attn_sink=True)
        eager = GQA(m)
        selected = GQA(m, global_attention="flash")
        selected.load_state_dict(eager.state_dict())
        x = torch.randn(2, 8, 16)
        torch.testing.assert_close(eager(x), selected(x), atol=0, rtol=0)

    def test_gate_and_initialization_are_preserved(self):
        m = copy.deepcopy(self.m)
        m.update(attn_gate=True, attn_gate_input="normalized",
                 attn_gate_bias=False, attn_gate_init="linear_default")
        torch.manual_seed(1234)
        eager = LanguageModel(m, self.rows, self.lengths)
        rng = torch.get_rng_state()
        torch.manual_seed(1234)
        sdpa = LanguageModel(m, self.rows, self.lengths, global_attention="sdpa")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(eager.state_dict()), list(sdpa.state_dict()))
        for key in eager.state_dict():
            self.assertTrue(torch.equal(eager.state_dict()[key], sdpa.state_dict()[key]))
        ids = torch.randint(0, 31, (2, 8))
        valid = torch.ones_like(ids, dtype=torch.bool)
        torch.testing.assert_close(eager(ids, valid), sdpa(ids, valid), atol=3e-6, rtol=3e-5)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            GQA(self.m, global_attention="unknown")

    def test_forced_flash_does_not_silently_fallback_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            GQA(self.m, global_attention="flash")(torch.randn(1, 8, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sdpa_repeat_and_finite_backward(self):
        old = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            def run():
                torch.manual_seed(1234)
                layer = GQA(self.m, global_attention="sdpa").cuda()
                x = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = layer(x)
                    loss = out.float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all().item())
                return out.detach(), x.grad
            a, ga = run()
            b, gb = run()
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(ga, gb))
        finally:
            torch.use_deterministic_algorithms(old)

    def test_sdpa_repeat_and_resume(self):
        def run(split=False):
            torch.manual_seed(1234)
            model = LanguageModel(self.m, self.rows, self.lengths, global_attention="sdpa")
            opt = torch.optim.AdamW(model.parameters(), lr=0.001)
            ids = torch.arange(16).reshape(2, 8) % 31
            valid = torch.ones_like(ids, dtype=torch.bool)
            trace = []
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                loss = model(ids, valid).float().square().mean()
                loss.backward()
                opt.step()
                trace.append(loss.item())
                if split and step == 1:
                    state = copy.deepcopy(model.state_dict())
                    optim = copy.deepcopy(opt.state_dict())
                    rng = torch.get_rng_state()
                    model = LanguageModel(self.m, self.rows, self.lengths, global_attention="sdpa")
                    model.load_state_dict(state)
                    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
                    opt.load_state_dict(optim)
                    torch.set_rng_state(rng)
            return trace, model.state_dict()
        a, sa = run()
        for split in (False, True):
            b, sb = run(split)
            self.assertEqual(a, b)
            self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))


class AttentionVariantsTests(unittest.TestCase):
    def test_global_sdpa_matches_eager_gqa(self):
        torch.manual_seed(1234)
        cfg, _, _ = fixture()
        m = cfg["model"]
        eager, sdpa = GQA(m), GQA(m, global_attention="sdpa")
        sdpa.load_state_dict(eager.state_dict())
        a = torch.randn(2, 8, 16, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        expected, actual = eager(a), sdpa(b)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(b.grad, a.grad, rtol=2e-5, atol=2e-6)
        for left, right in zip(eager.parameters(), sdpa.parameters()):
            torch.testing.assert_close(right.grad, left.grad, rtol=2e-5, atol=2e-6)

    def test_sdpa_leaves_local_attention_with_sink_on_eager(self):
        cfg, _, _ = fixture()
        m = cfg["model"]
        m.update(attn_pattern=["L", "G"], attn_window=4, attn_sink=True)
        layer = GQA(m, global_attention="sdpa")
        self.assertEqual(layer.attention_backend, "eager")
        self.assertIsNotNone(layer.sink)


if __name__ == "__main__":
    unittest.main()
