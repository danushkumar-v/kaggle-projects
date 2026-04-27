# 01 — CLOVER Demo: Class-Overlapping Continual Learning

> A guided demo showing how the [CLOVER](https://github.com/danushkumar-v/clover-cl)
> benchmark library exposes class-revisit scenarios on CIFAR-100, evaluated
> with a simple replay baseline.

## What this project shows

- Three CL scenarios on the same dataset: disjoint, partial overlap, exact replay
- Both APIs of CLOVER side-by-side: `OverlapDataManager` (PILOT-compatible)
  and `build_benchmark(StreamSpec(...))` (declarative, multi-seed friendly)
- The killer metric: forgetting on **revisiting** vs. **fresh** classes
- End-to-end training loop that runs in under 5 minutes on a Kaggle T4

## Run

```bash
make run-local      # run notebook locally
make push           # push to Kaggle (re-executes there)
make push-fast      # push without re-execution
make pull           # pull latest version
make status         # check execution status
make open           # print Kaggle URL
```

## Notebook structure

| § | Topic |
|---|-------|
| 1 | Introduction |
| 2 | The CLOVER library |
| 3 | Designing an overlap stream |
| 4 | Dataset (Split-CIFAR-100) |
| 5 | Baseline — Replay |
| 6 | Training across three scenarios |
| 7 | Results and analysis |
| 8 | Discussion |
| 9 | Conclusion |
| 10 | References |

## Compute

Designed for Kaggle's free T4 GPU. Total notebook runtime target: ~5 minutes.
Falls back to CPU if needed (slower; 15–20 min).

## License

MIT — same as the parent repo.
