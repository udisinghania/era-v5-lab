# Global attention follow-up to #536

## Scope and provenance

Restores the global attention implementation and its original attention tests from
#536 commit `91e00cd`, onto main `f09d424` (merged #536: `c4df253`).
Only the attention portion is extracted: no sparse expert dispatch, padding trimming,
optimizer changes, reversibility or distributed training.

`--global-attention eager|sdpa|flash` defaults to **eager**. Global SDPA uses
compact KV heads and PyTorch `enable_gqa`; forced flash restricts the SDPA backend
to `FLASH_ATTENTION`. Unsupported forced-flash requests fail rather than silently
falling back. Local sliding-window/sink layers stay eager. Existing gates and RoPE
are preserved. There is no new flash-attn package dependency.

The archived kernel code is reused, with explicit invalid-choice and CPU-flash
request guards. The requested implementation and effective per-layer requests are
recorded in the run identity, so changing them invalidates checkpoint resume.
SDPA is automatic backend selection: the manifest does not claim it selects Flash.
No config_v1, runtime contract or lead-owned arm is edited.

## Verification, 21 November 2026

- Full G5 pytest suite: **168 passed, 4 skipped**, 56.49 s.
- Original #536 attention tests retained, plus causality, GQA/MQA/MHA output and
  gradient agreement, partial RoPE, gates/initialization, local/sink preservation,
  deterministic CUDA SDPA backward, and model/optimizer split-resume tests.
- Two independent seed-1234 CUDA BF16 T2-Muon smoke runs: 5 steps, context 128,
  batch 2, local tiny text fixtures; losses.txt byte-identical and final model hashes
  identical. This is execution evidence, not official T2 quality evidence.
- Final model SHA256: `1c27461454fa8cbb1a0c774e4e91d2ccb93749345b67dd902b974ec9239c9f38`.
- First CE 12.031235694885254, last CE 1.0725690126419067.
- First run training/evaluation wall time: 6.972626700000546 s.
- SDPA and eager traces differ; neither bitwise cross-backend equivalence nor a
  held-out quality winner is claimed.

## Local synthetic attention-only timing

RTX 3070 Laptop, PyTorch 2.5.1, BF16, width512/8Q/2KV, sequence1024, batch2.
Three warmup iterations and ten timed forward/backward iterations per backend;
CUDA synchronized at measurement boundaries. Ratio is eager median / candidate
median. This is attention-only timing, not end-to-end training step timing.

| Path | Median ms | Before/after | Observed backend |
| --- | ---: | ---: | --- |
| eager | 9.666 | 1.000 | explicit eager |
| sdpa | 12.121 | 0.797 | SDPA math |
| forced flash | unavailable | unavailable | not compiled in this Windows build |

SDPA BF16 relative L2 error against eager: output 0.003235,
input gradient 0.003537, weight gradient 0.003498.
The local math backend is slower; this PR does not claim a local fused-kernel gain.

Reproduce the kernel screen from the repository root:

```bash
python teams/05-architecture/notebooks/benchmark_global_attention.py --device cuda --seq-len 1024 --batch-size 2 --repeats 10 --out out/attention_ab.json
python -m pytest teams/05-architecture/tests -q
```

## Target-GPU acceptance still required

TSAI's previous official #536 screen reported 1.098x for global SDPA and 1.079x
for forced Flash, with changed traces. Those historical numbers are not fresh
measurements of this branch. See [the official result](https://github.com/theschoolofai/era-v5-lab/pull/536#issuecomment-7033189262).

Before considering any default change: run matched current eager/SDPA/forced-flash
commands on the supported GPU with the same arm, seed, precision, data, global and
microbatch sizes. Record the actual selected kernels, warmup-excluded median step
time, ratio, memory, output/gradient errors, per-step loss differences and held-out
BPB differences. Check deterministic repeat and checkpoint/resume on each selected
backend. Numerical acceptance and any default change remain with TSAI/G5 lead.
