# 02 -- L2P + class overlap: when CL classes come back

> Running [L2P](https://arxiv.org/abs/2112.08654) (Learning to Prompt,
> CVPR 2022) on two CIFAR-100 task streams designed with the
> [CLOVER](https://github.com/danushkumar-v/clover-cl) benchmark library:
> a partial-overlap stream and a full-replay stream. The headline question:
> what does the prompt pool actually do when classes come back?

## What this project shows

- A clean, ~80-line implementation of L2P on a frozen ViT-B/16
  (timm, ImageNet-21k pretrained)
- Two carefully constructed CIFAR-100 streams using the CLOVER
  `OverlapSpec` API: partial overlap (5 recurring classes, each in two
  tasks) and exact overlap (task 0 == task 2 in classes, image-disjoint
  samples)
- Standard CL metrics (accuracy matrix, BWT, FWT) plus a per-class
  recurring-vs-novel breakdown and three prompt-pool diagnostics that
  open the black box on what the prompts actually learned
- A short failure-mode tour using real misclassified images

## Setup

- **Backbone:** ViT-B/16 pretrained on ImageNet-21k (frozen)
- **Method:** L2P -- inline implementation, no submodule hopping
- **Dataset:** CIFAR-100, 4 tasks, 10 classes per task

## Run

```bash
make push           # push to Kaggle (re-executes there on T4)
make status         # check Kaggle execution status
make pull           # pull executed notebook back (with outputs)
make open           # print Kaggle URL
```

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

## Compute

Designed for Kaggle's free T4 GPU. Total runtime target: 60-90 minutes
(two training runs at ~25 min each plus eval and plotting).

## License

MIT.
