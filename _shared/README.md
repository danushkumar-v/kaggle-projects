# Shared utilities

This directory holds code reused across multiple projects:

- `styling.py` — markdown templates and color palettes (added in Step C)
- `plot_themes.py` — matplotlib/seaborn theme presets
- `kaggle_helpers.py` — common Kaggle path resolution, dataset loading

Import from notebooks via:

```python
import sys
sys.path.insert(0, '../../_shared')
import styling
```

Or, when running on Kaggle, copy the needed file to the working dir
(handled automatically by the project's notebook setup cell).
