# Kaggle Projects

A small collection of Kaggle notebooks on continual learning, vision
transformers, and benchmarking. Each project is one folder, one
notebook, designed to read end-to-end and run reproducibly on Kaggle's
free T4 GPU.

## Projects

| # | Project | What it shows | Kaggle |
|---|---------|---------------|--------|
| 01 | [clover-demo-cl-benchmark](projects/01-clover-demo-cl-benchmark) | Three CL scenarios (disjoint / partial / exact) on CIFAR-100 with a replay baseline | [link](https://www.kaggle.com/code/danushkumarv/clover-class-overlap-in-continual-learning) |
| 02 | [l2p-clover-overlap-analysis](projects/02-l2p-clover-overlap-analysis) | L2P prompt-based CL on overlapping CIFAR-100 streams + prompt-pool diagnostics | [link](https://www.kaggle.com/code/danushkumarv/l2p-class-overlap-when-cl-classes-come-back) |

The Kaggle link is the live version with executed outputs. The
notebooks in this repo are the source of truth.

## Layout

```
kaggle-projects/
  _shared/      shared helpers (styling, plotting)
  projects/     one folder per notebook project
    NN-slug/
      notebook.ipynb
      kernel-metadata.json
      README.md
      Makefile
  scripts/      workspace-level scripts
```

## Running a notebook locally

Each project ships a Makefile with the same targets:

```bash
cd projects/02-l2p-clover-overlap-analysis
make run-local       # open in local Jupyter
make push            # push to Kaggle (re-executes on T4)
make pull            # pull latest executed notebook from Kaggle
make status          # check Kaggle execution status
make open            # print the Kaggle URL
```

To push to Kaggle you'll need the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
authenticated via `~/.kaggle/kaggle.json`.

## Conventions

- Notebooks are committed **with outputs** -- they're the published
  artifact.
- Datasets are fetched at runtime, never committed.
- One palette per project, locked at the top of each notebook for
  visual consistency.

## License

MIT.
