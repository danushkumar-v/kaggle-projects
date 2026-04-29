# Kaggle Projects

Curated collection of Kaggle notebook projects — research demos, ML
experiments, and benchmark studies. Every project is one folder, one
notebook, with reproducible local + Kaggle execution.

## Layout

```
kaggle-projects/
├── _shared/              shared helpers (styling, plotting, utils)
├── _templates/           scaffolding templates for new projects
├── projects/             one folder per notebook project
│   └── 01-clover-demo-cl-benchmark/
│       ├── notebook.ipynb
│       ├── kernel-metadata.json
│       ├── README.md
│       └── Makefile
├── scripts/              workspace-level scripts
└── Makefile              workspace-level commands
```

## Quick start

### One-time setup

1. Install Kaggle CLI: `pip install kaggle`
2. Get your API token from https://www.kaggle.com/settings → "Create New Token"
3. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`
5. Verify: `make check-kaggle`

### Working on a project

```bash
cd projects/01-clover-demo-cl-benchmark
make run-local       # run the notebook locally
make push            # push to Kaggle (re-executes there)
make push-fast       # push without re-execution (preserves local outputs)
make pull            # pull latest version + outputs from Kaggle
make status          # check Kaggle execution status
```

### Creating a new project

```bash
./scripts/new_project.sh <project-slug>
```

This scaffolds a new folder under `projects/` with a notebook, metadata,
README, and Makefile pre-wired.

## Project index

| # | Project | Topic | Kaggle |
|---|---------|-------|--------|
| 01 | clover-demo-cl-benchmark | Continual learning with class overlap | https://www.kaggle.com/code/danushkumarv/clover-class-overlap-in-continual-learning |
| 02 | l2p-clover-overlap-analysis | L2P prompt-based CL on overlapping CIFAR-100 streams | https://www.kaggle.com/code/danushkumarv/l2p-class-overlap-when-cl-classes-come-back |

## Notes

- Notebooks are committed **with outputs** — they're the published artifact.
- `kaggle.json` is `.gitignore`'d. Never commit it.
- Datasets larger than 50MB are not committed; they're fetched at runtime
  via Kaggle's dataset API or the source URL in the notebook.
