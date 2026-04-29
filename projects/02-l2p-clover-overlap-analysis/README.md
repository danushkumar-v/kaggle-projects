---
project: 02-l2p-clover-overlap-analysis
palette: forest-sage (id 2)
---

# 02 -- L2P + class overlap: when CL classes come back

> Running L2P (Learning to Prompt, CVPR 2022) on two CIFAR-100 task
> streams designed with the [CLOVER](https://github.com/danushkumar-v/clover-cl)
> benchmark library: a partial-overlap stream and a full-replay stream.
> The headline question: what does the prompt pool actually do when
> classes come back?

## Setup

- **Backbone:** ViT-B/16 pretrained on ImageNet-21k (frozen)
- **Method:** L2P -- inline ~80-line implementation, no submodule hopping
- **Dataset:** CIFAR-100, 4 tasks, 10 classes per task
- **Scenario A (partial):** 5 classes recur across two tasks each (image-disjoint)
- **Scenario B (exact):** task 0 == task 2 in classes (image-disjoint samples)

## Run

```bash
make push           # push to Kaggle as PRIVATE (re-executes there on T4)
make status         # check Kaggle execution status
make pull           # pull executed notebook back (with outputs)
make open           # print Kaggle URL
```

The Kaggle T4 re-run is the authoritative one. Local execution is not
required -- the notebook commits with code, and outputs are produced
on Kaggle's runners.

## Notebook structure

| S | Topic |
|---|-------|
| 1 | A simple question |
| 2 | Setup |
| 3 | The two scenarios (visualized) |
| 4 | L2P in 80 lines |
| 5 | Training run 1 -- partial overlap |
| 6 | Training run 2 -- exact overlap |
| 7 | The accuracy matrix |
| 8 | Forgetting and backward transfer |
| 9 | Recurring vs novel classes |
| 10 | What did the prompts learn? |
| 11 | Failure mode tour |
| 12 | Closing thoughts |

## Compute budget

Targets Kaggle's free T4. Full schedule:

- L2P training scenario A: ~25 min
- L2P training scenario B: ~25 min
- Eval + plots: ~15 min
- Total: ~65 min (under the 12-hour Kaggle limit)

## Tags to add on Kaggle (after manual flip to public)

Paste these into the notebook settings on Kaggle:

- continual learning
- vision transformer
- transfer learning
- pytorch
- image classification
- deep learning
- tutorial
- cifar-100

## Notebook description for the Kaggle card

> L2P (CVPR 2022) is one of the most popular prompt-based continual
> learning methods. It works well on disjoint task splits -- but what
> happens when classes come back? Using the CLOVER benchmark, I run
> L2P on two overlap scenarios and look at what the prompt pool
> actually learns. Spoiler in S9.

## License

MIT.
