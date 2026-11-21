"""Synthetic attention-only A/B screen, not a model-quality experiment.

Run from any directory. Includes forward/backward and CUDA synchronization at
measurement boundaries. Forced flash never silently falls back to math.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from week2.model import GQA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("out/attention_ab.json"))
    args = p.parse_args()
    if min(args.seq_len, args.batch_size, args.repeats) < 1:
        p.error("sequence length, batch size and repeats must be positive")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    m = dict(n_embd=512, n_head=8, n_kv_head=2, n_layer=8, rope_theta=10000.0)
    baseline = GQA(m).to(device)
    initial = copy.deepcopy(baseline.state_dict())
    x = torch.randn(args.batch_size, args.seq_len, 512, device=device)
    results = {}
    reference = None

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for backend in ("eager", "sdpa", "flash"):
        layer = GQA(m, global_attention=backend).to(device)
        layer.load_state_dict(initial)
        inp = x.detach().clone().requires_grad_()

        def step():
            layer.zero_grad(set_to_none=True)
            inp.grad = None
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = layer(inp)
                loss = output.float().square().mean()
            loss.backward()
            return output

        try:
            for _ in range(3):
                step()
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeats):
                sync()
                start = time.perf_counter()
                output = step()
                sync()
                times.append(1000 * (time.perf_counter() - start))
            tensors = (output.detach().float(), inp.grad.detach().float(),
                       torch.cat([v.grad.detach().flatten().float() for v in layer.parameters()]))
            if reference is None:
                reference = tuple(t.clone() for t in tensors)
            errors = {}
            for name, ref, value in zip(("output", "input_grad", "weight_grad"), reference, tensors):
                errors[name] = dict(max_abs=(ref - value).abs().max().item(),
                                    relative_l2=((ref - value).norm() / ref.norm().clamp_min(1e-30)).item())
            # Profile separately: profiling overhead is excluded from timing.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                step()
            operators = sorted({e.key for e in prof.key_averages()
                                if "scaled_dot_product" in e.key or "flash_attention" in e.key})
            results[backend] = dict(median_ms=statistics.median(times), errors=errors,
                                    observed_attention_ops=operators,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
        except RuntimeError as exc:
            if backend != "flash":
                raise
            results[backend] = dict(unavailable=str(exc))
    before = results["eager"]["median_ms"]
    for value in results.values():
        if "median_ms" in value:
            value["before_over_after"] = before / value["median_ms"]
    report = dict(scope="SYNTHETIC_ATTENTION_ONLY_NOT_QUALITY", torch=torch.__version__,
                  device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                  seq_len=args.seq_len, batch_size=args.batch_size, repeats=args.repeats,
                  seed=1234, dtype="bf16" if device.type == "cuda" else "fp32",
                  results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
