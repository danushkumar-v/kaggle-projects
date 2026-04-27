import json

NB = 'projects/01-clover-demo-cl-benchmark/notebook.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# ── Fix 1: install cell — --no-deps to avoid replacing Kaggle's GPU torch ──
NEW_INSTALL = """\
import subprocess, sys

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)

# Install clover WITHOUT pulling in its own torch/torchvision deps.
# Kaggle's pre-installed torch is GPU-optimised; letting pip replace it can
# produce a build whose CUDA kernels don't match the GPU's compute capability.
_pip("--no-deps", "git+https://github.com/danushkumar-v/clover-cl.git")
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'subprocess' in ''.join(cell['source']):
        cell['source'] = NEW_INSTALL
        print('Fixed: install cell')
        break

# ── Fix 2: DEVICE — add kernel-launch sanity check ─────────────────────────
OLD = 'DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"'
NEW = '''\
if torch.cuda.is_available():
    try:
        _t = torch.zeros(2, 2).cuda()
        _t.add_(1)  # verify a real kernel launches
        del _t
        DEVICE = "cuda"
    except Exception as _e:
        print(f"WARNING: CUDA kernel launch failed ({_e}). Falling back to CPU.")
        DEVICE = "cpu"
else:
    DEVICE = "cpu"'''

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if OLD in src:
            new_src = src.replace(OLD, NEW)
            # also add cudnn.benchmark after manual_seed
            new_src = new_src.replace(
                'torch.manual_seed(SEED)',
                'torch.manual_seed(SEED)\nif DEVICE == "cuda":\n    torch.backends.cudnn.benchmark = True'
            )
            cell['source'] = new_src.splitlines(keepends=True)
            print('Fixed: DEVICE sanity check + cudnn.benchmark')
            break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved.')
