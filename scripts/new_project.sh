#!/usr/bin/env bash
# Scaffold a new project folder.
# Usage: ./scripts/new_project.sh <slug>
#   slug examples: 02-titanic-eda, 03-arxiv-text-classification

set -e

SLUG="$1"
if [ -z "$SLUG" ]; then
  echo "Usage: $0 <slug>"
  echo "Example: $0 02-titanic-eda"
  exit 1
fi

DIR="projects/$SLUG"
if [ -d "$DIR" ]; then
  echo "❌ $DIR already exists"
  exit 1
fi

mkdir -p "$DIR"

cat > "$DIR/README.md" <<EOF
# $SLUG

> One-line description goes here.

## Run

\`\`\`bash
make run-local       # run notebook in local Jupyter
make push            # push to Kaggle (re-executes)
make push-fast       # push without re-execution
make pull            # pull latest from Kaggle
make status          # check Kaggle execution status
\`\`\`

## Notes
- Replace this section with project-specific context.
EOF

cat > "$DIR/kernel-metadata.json" <<EOF
{
  "id": "REPLACE_WITH_KAGGLE_USERNAME/$SLUG",
  "title": "$SLUG",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

cat > "$DIR/Makefile" <<'EOF'
.PHONY: run-local push push-fast pull status open

NB := notebook.ipynb

run-local:
	jupyter notebook $(NB)

push:
	kaggle kernels push -p .

push-fast:
	kaggle kernels push -p . --no-execute || \
	  (echo "Note: --no-execute may not be supported by your kaggle CLI version. Falling back."; \
	   kaggle kernels push -p .)

pull:
	kaggle kernels pull -p . -m

status:
	@kid=$$(jq -r .id kernel-metadata.json); \
	kaggle kernels status $$kid

open:
	@kid=$$(jq -r .id kernel-metadata.json); \
	echo "https://www.kaggle.com/code/$$kid"
EOF

cat > "$DIR/notebook.ipynb" <<EOF
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# $SLUG\n", "\n", "(Replace this with the project's first markdown cell.)\n"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": []}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF

echo "✅ Scaffolded $DIR"
echo "   Next:"
echo "     1. Edit $DIR/kernel-metadata.json — set the 'id' field to your Kaggle slug"
echo "     2. cd $DIR && jupyter notebook $NB"
