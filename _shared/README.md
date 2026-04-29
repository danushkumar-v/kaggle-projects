# Shared

Reserved for code that's genuinely reused across more than one project.

The notebooks in this repo are intentionally self-contained: plot
theming, palette colors, and small helpers are inlined at the top of
each notebook rather than imported from here. That keeps each notebook
readable as a standalone artifact on Kaggle, where users can fork and
run without dragging in an unfamiliar package layout.

If a shared module is added in the future it'll be importable via:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '_shared'))
```
