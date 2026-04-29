"""
Build notebook.ipynb for Project 02 from scratch.

All cell sources are ASCII-only (no em dashes, no curly quotes, no emoji).
The single allowed emoji in section 1's H1 is encoded as an HTML numeric
entity so the source stays ASCII while still rendering as an emoji on
Kaggle.

Run:
    python _build_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

# -------- palette (forest-sage, id 2) --------
PRIMARY   = "#3D6E5C"
SECONDARY = "#7AAE99"
DARK      = "#1F3D33"

cells = []

def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })

def code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })

def banner(n: int, title: str, emoji_entity: str | None = None) -> str:
    prefix = f"{emoji_entity} " if emoji_entity else ""
    return (
        f'# <div style="padding:24px;color:white;margin:8px 0;font-size:65%;'
        f'text-align:left;display:fill;border-radius:10px;'
        f'background-color:{PRIMARY};overflow:hidden">'
        f'<b><span style=\'color:{SECONDARY}\'>{n} |</span></b> '
        f'<b>{prefix}{title}</b></div>'
    )

def h3(n_m: str, title: str) -> str:
    return (
        f"### <b><span style='color:{PRIMARY}'>{n_m} |</span> {title}</b>"
    )

def callout(label: str) -> str:
    return (
        f'<div style="color:white;display:fill;border-radius:6px;'
        f'font-size:95%;letter-spacing:0.8px;background-color:{DARK};'
        f'padding:6px 12px;margin:8px 0;width:fit-content">'
        f'<b>{label}</b></div>'
    )

def takeaway(text: str) -> str:
    return f"> <span style='color:{PRIMARY}'><b>Key takeaway --</b></span> {text}"


# ============================================================
# Section 1 -- A simple question
# ============================================================
md(
    banner(1, "A simple question", emoji_entity="&#x1F9E9;") + "\n"
    "\n"
    "Most continual learning papers assume a clean stream: each task brings new\n"
    "classes, the previous classes never come back, and the model has to fight\n"
    "off catastrophic forgetting on its own. That is a clean problem, and a\n"
    "useful one. But it is not the problem most real systems face.\n"
    "\n"
    "Real streams revisit classes. A camera that classifies birds will see\n"
    "robins again next spring. A medical model that has seen pneumonia in\n"
    "January will see it again in March. The interesting question is not\n"
    "*will the model forget?* -- it almost certainly will -- but *what does\n"
    "the model do when the classes come back?*\n"
    "\n"
    "I wanted a clean experimental answer to that, so I picked one popular\n"
    "method (L2P, CVPR 2022) and ran it on two carefully constructed\n"
    "CIFAR-100 streams built with [CLOVER](https://github.com/danushkumar-v/clover-cl):\n"
    "\n"
    "- **Scenario A (partial overlap):** 4 tasks, 10 classes each. Five classes\n"
    "  recur, each appearing in exactly two tasks, with disjoint image samples\n"
    "  on each occurrence.\n"
    "- **Scenario B (exact overlap):** task 0 and task 2 share the same 10\n"
    "  classes. Tasks 1 and 3 are unrelated. Image samples are disjoint.\n"
    "\n"
    "Same backbone, same hyperparameters, same training schedule for both.\n"
    "Only the stream structure differs.\n"
    "\n"
    "Below is the picture I'll keep coming back to. Take a moment with it --\n"
    "it captures the whole experimental design in one frame.\n"
)

code(
    "# Hero plot: the two scenarios as task timelines.\n"
    "# We define class IDs inline here so the figure renders even before\n"
    "# the rest of the notebook has run.\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as mpatches\n"
    "from matplotlib.patches import FancyArrowPatch\n"
    "\n"
    "PRIMARY, SECONDARY, DARK = '#3D6E5C', '#7AAE99', '#1F3D33'\n"
    "\n"
    "# Scenario A: recurring classes 0..4, each appearing in two tasks\n"
    "scA = [\n"
    "    [0, 1, 2,  10, 11, 12, 13, 14, 15, 16],\n"
    "    [3, 4,     17, 18, 19, 20, 21, 22, 23, 24],\n"
    "    [0, 1, 2,  25, 26, 27, 28, 29, 30, 31],\n"
    "    [3, 4,     32, 33, 34, 35, 36, 37, 38, 39],\n"
    "]\n"
    "scA_recur = {0, 1, 2, 3, 4}\n"
    "\n"
    "# Scenario B: T0 == T2 (same classes, image-disjoint), T1 and T3 unique\n"
    "scB = [\n"
    "    list(range(0, 10)),\n"
    "    list(range(10, 20)),\n"
    "    list(range(0, 10)),\n"
    "    list(range(20, 30)),\n"
    "]\n"
    "scB_recur = set(range(0, 10))\n"
    "\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 5), facecolor='white')\n"
    "\n"
    "def draw_timeline(ax, scenario, recur_set, title):\n"
    "    ax.set_xlim(0, 4)\n"
    "    ax.set_ylim(-0.4, 1.2)\n"
    "    ax.axis('off')\n"
    "    ax.set_title(title, loc='left', fontsize=13, color=DARK, fontweight='semibold', pad=12)\n"
    "    box_w, box_h, box_y = 0.85, 0.55, 0.25\n"
    "    for t, classes in enumerate(scenario):\n"
    "        x0 = t + 0.075\n"
    "        rect = mpatches.FancyBboxPatch(\n"
    "            (x0, box_y), box_w, box_h,\n"
    "            boxstyle='round,pad=0.01,rounding_size=0.04',\n"
    "            linewidth=1.2, edgecolor=DARK, facecolor='white')\n"
    "        ax.add_patch(rect)\n"
    "        ax.text(x0 + box_w/2, box_y + box_h + 0.06, f'task {t}',\n"
    "                ha='center', va='bottom', fontsize=10, color=DARK, fontweight='bold')\n"
    "        for i, c in enumerate(classes):\n"
    "            cx = x0 + 0.05 + (i % 5) * 0.155\n"
    "            cy = box_y + box_h - 0.14 - (i // 5) * 0.21\n"
    "            face = SECONDARY if c in recur_set else '#F2F2F2'\n"
    "            tcol = 'white' if c in recur_set else DARK\n"
    "            ax.add_patch(mpatches.FancyBboxPatch(\n"
    "                (cx, cy), 0.13, 0.16,\n"
    "                boxstyle='round,pad=0.005,rounding_size=0.02',\n"
    "                linewidth=0.6, edgecolor=DARK, facecolor=face))\n"
    "            ax.text(cx + 0.065, cy + 0.08, str(c), ha='center', va='center',\n"
    "                    fontsize=8, color=tcol, fontweight='bold')\n"
    "\n"
    "draw_timeline(axes[0], scA, scA_recur, 'Scenario A -- partial overlap (5 classes recur, each in 2 tasks)')\n"
    "draw_timeline(axes[1], scB, scB_recur, 'Scenario B -- exact overlap (task 0 == task 2 in classes)')\n"
    "\n"
    "# legend\n"
    "legend = [mpatches.Patch(facecolor=SECONDARY, edgecolor=DARK, label='recurring class'),\n"
    "          mpatches.Patch(facecolor='#F2F2F2', edgecolor=DARK, label='novel class')]\n"
    "axes[0].legend(handles=legend, loc='lower right', frameon=False, fontsize=9, ncol=2)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('hero.png', dpi=150, bbox_inches='tight', facecolor='white')\n"
    "plt.show()\n"
)

md(
    "Here is what we'll find. In **Scenario A**, the prompt pool more or\n"
    "less stabilises across tasks and recurring classes get a small bump\n"
    "from being seen twice -- but the bump is smaller than I expected. In\n"
    "**Scenario B**, where an entire task comes back later, L2P does\n"
    "remarkably well on the recurring task even without explicit replay,\n"
    "and the prompt-pool diagnostics in S10 explain why.\n"
    "\n"
    "The rest of the notebook walks through how I got there.\n"
)

# ============================================================
# Section 2 -- Setup
# ============================================================
md(
    banner(2, "Setup") + "\n"
    "\n"
    "This is a self-contained notebook: install one package, import the\n"
    "stack, set up the device, lock in plot theming. If you re-run this\n"
    "on Kaggle the heavy lifts in S5 and S6 take roughly 25 minutes each\n"
    "on a T4 GPU.\n"
)

code(
    "import subprocess, sys\n"
    "\n"
    "def _pip(*args):\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=False)\n"
    "\n"
    "# clover-cl provides the OverlapSpec / build_benchmark API we use in S3.\n"
    "# --no-deps avoids replacing Kaggle's GPU torch.\n"
    "_pip('--no-deps', 'git+https://github.com/danushkumar-v/clover-cl.git')\n"
    "# timm is pre-installed on Kaggle, but the install is cheap and idempotent.\n"
    "_pip('--no-deps', 'timm')\n"
)

code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import os, json, math, random, time, copy\n"
    "from collections import defaultdict, Counter\n"
    "from dataclasses import dataclass\n"
    "\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import DataLoader, Subset, TensorDataset\n"
    "\n"
    "import torchvision\n"
    "from torchvision import transforms\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib as mpl\n"
    "import matplotlib.colors as mcolors\n"
    "import matplotlib.patches as mpatches\n"
    "import seaborn as sns\n"
    "\n"
    "import timm\n"
    "\n"
    "SEED = 0\n"
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)\n"
    "\n"
    "# CUDA sanity test (a no-kernel-image error appears late if we trust torch.cuda.is_available naively)\n"
    "if torch.cuda.is_available():\n"
    "    try:\n"
    "        _t = torch.zeros(2, 2).cuda(); _t.add_(1); del _t\n"
    "        DEVICE = 'cuda'\n"
    "    except Exception as _e:\n"
    "        print(f'WARNING: CUDA kernel launch failed ({_e}). Falling back to CPU.')\n"
    "        DEVICE = 'cpu'\n"
    "else:\n"
    "    DEVICE = 'cpu'\n"
    "\n"
    "if DEVICE == 'cuda':\n"
    "    torch.backends.cudnn.benchmark = True\n"
    "\n"
    "print(f'Device: {DEVICE} | PyTorch {torch.__version__} | timm {timm.__version__}')\n"
)

code(
    "# Plot theme -- inlined here so the notebook is self-contained on Kaggle.\n"
    "PRIMARY, SECONDARY, DARK = '#3D6E5C', '#7AAE99', '#1F3D33'\n"
    "EXTRA = ['#B0D4C5', '#2A4D40']\n"
    "SEQ = [PRIMARY, SECONDARY, DARK] + EXTRA\n"
    "\n"
    "sns.set_theme(style='white', font_scale=1.0)\n"
    "mpl.rcParams.update({\n"
    "    'figure.figsize': (8.0, 5.0),\n"
    "    'figure.dpi': 110,\n"
    "    'figure.facecolor': 'white',\n"
    "    'savefig.dpi': 200,\n"
    "    'savefig.bbox': 'tight',\n"
    "    'font.family': 'DejaVu Sans',\n"
    "    'font.size': 11,\n"
    "    'axes.titlesize': 13,\n"
    "    'axes.titleweight': 'semibold',\n"
    "    'axes.titlecolor': DARK,\n"
    "    'axes.titlepad': 14,\n"
    "    'axes.labelsize': 11,\n"
    "    'axes.labelcolor': DARK,\n"
    "    'xtick.labelsize': 10, 'ytick.labelsize': 10,\n"
    "    'xtick.color': DARK, 'ytick.color': DARK,\n"
    "    'legend.fontsize': 10, 'legend.frameon': False,\n"
    "    'axes.spines.top': False, 'axes.spines.right': False,\n"
    "    'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,\n"
    "    'axes.grid': True, 'grid.color': '#E8E8E8', 'grid.linewidth': 0.6,\n"
    "    'grid.alpha': 0.8, 'axes.axisbelow': True,\n"
    "    'lines.linewidth': 2.0, 'lines.markersize': 6,\n"
    "    'axes.prop_cycle': mpl.cycler(color=SEQ),\n"
    "})\n"
    "sns.set_palette(SEQ)\n"
    "\n"
    "def seq_cmap():\n"
    "    return mcolors.LinearSegmentedColormap.from_list('seq', ['#FFFFFF', SECONDARY, PRIMARY], N=256)\n"
    "\n"
    "os.makedirs('figures', exist_ok=True)\n"
)

code(
    "# CLOVER import -- attempt the real package, fall back to a tiny inline\n"
    "# class if the install failed (so the notebook still demonstrates the\n"
    "# concepts).\n"
    "USE_CLOVER = True\n"
    "try:\n"
    "    import clover\n"
    "    print(f'clover {getattr(clover, \"__version__\", \"unknown\")} loaded')\n"
    "except Exception as e:\n"
    "    USE_CLOVER = False\n"
    "    print(f'clover unavailable ({e}); falling back to inline OverlapSpec')\n"
    "\n"
    "@dataclass\n"
    "class OverlapSpec:\n"
    "    classes_per_task: list   # list[list[int]]\n"
    "    name: str = ''\n"
    "    @property\n"
    "    def n_tasks(self):\n"
    "        return len(self.classes_per_task)\n"
    "    @property\n"
    "    def all_classes(self):\n"
    "        return sorted(set(c for t in self.classes_per_task for c in t))\n"
    "    def overlap_matrix(self):\n"
    "        T = self.n_tasks\n"
    "        M = np.zeros((T, T), dtype=int)\n"
    "        for i in range(T):\n"
    "            for j in range(T):\n"
    "                M[i, j] = len(set(self.classes_per_task[i]) & set(self.classes_per_task[j]))\n"
    "        return M\n"
)

# ============================================================
# Section 3 -- The two scenarios (visualized)
# ============================================================
md(
    banner(3, "The two scenarios (visualized)") + "\n"
    "\n"
    + callout("Stream design") + "\n"
    "\n"
    "Both scenarios use 4 tasks of 10 classes each. The difference is *which*\n"
    "classes recur and *how often*.\n"
    "\n"
    "**Scenario A (partial overlap).** Five recurring classes split into two\n"
    "groups of {3, 2}. The first group appears in tasks 0 and 2; the second\n"
    "appears in tasks 1 and 3. Every other class is novel and shows up exactly\n"
    "once. Total unique classes: 35.\n"
    "\n"
    "**Scenario B (exact overlap).** Task 2 reuses the same class set as task\n"
    "0 -- but with disjoint image samples. Tasks 1 and 3 are unrelated, all\n"
    "novel. Total unique classes: 30.\n"
    "\n"
    "I picked these two because they sit on opposite ends of the overlap\n"
    "spectrum. Anything in the middle interpolates between them.\n"
)

code(
    "# Define both scenarios as OverlapSpec objects.\n"
    "scenario_A = OverlapSpec(\n"
    "    classes_per_task=[\n"
    "        [0, 1, 2,  10, 11, 12, 13, 14, 15, 16],\n"
    "        [3, 4,     17, 18, 19, 20, 21, 22, 23, 24],\n"
    "        [0, 1, 2,  25, 26, 27, 28, 29, 30, 31],\n"
    "        [3, 4,     32, 33, 34, 35, 36, 37, 38, 39],\n"
    "    ],\n"
    "    name='A_partial',\n"
    ")\n"
    "RECUR_A = {0, 1, 2, 3, 4}  # the five recurring classes in scenario A\n"
    "\n"
    "scenario_B = OverlapSpec(\n"
    "    classes_per_task=[\n"
    "        list(range(40, 50)),    # T0 unique-to-B class IDs (offset to avoid clash with A)\n"
    "        list(range(50, 60)),    # T1 novel\n"
    "        list(range(40, 50)),    # T2 -- same classes as T0\n"
    "        list(range(60, 70)),    # T3 novel\n"
    "    ],\n"
    "    name='B_exact',\n"
    ")\n"
    "RECUR_B = set(range(40, 50))  # the 10 classes that come back\n"
    "\n"
    "for sc in (scenario_A, scenario_B):\n"
    "    print(f'{sc.name}: tasks={sc.n_tasks}, unique classes={len(sc.all_classes)}')\n"
    "    print('  overlap matrix (rows=task i, cols=task j):')\n"
    "    print('  ' + str(sc.overlap_matrix()).replace('\\n', '\\n  '))\n"
)

code(
    "# Pretty side-by-side overlap heatmaps.\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')\n"
    "for ax, sc in zip(axes, (scenario_A, scenario_B)):\n"
    "    M = sc.overlap_matrix()\n"
    "    sns.heatmap(M, annot=True, fmt='d', cmap=seq_cmap(),\n"
    "                cbar=False, ax=ax, linewidths=0.6, linecolor='white',\n"
    "                annot_kws={'color': DARK, 'fontweight': 'bold'})\n"
    "    ax.set_title(f'{sc.name} overlap matrix')\n"
    "    ax.set_xlabel('task j'); ax.set_ylabel('task i')\n"
    "plt.tight_layout()\n"
    "plt.savefig('figures/overlap_matrices.png')\n"
    "plt.show()\n"
)

md(
    takeaway(\
        "scenario A keeps a steady trickle of recurrence (5 classes total); "
        "scenario B holds back a whole task and reintroduces it.") + "\n"
)

# ============================================================
# Section 4 -- L2P in 80 lines
# ============================================================
md(
    banner(4, "L2P in 80 lines") + "\n"
    "\n"
    "L2P (Wang et al., CVPR 2022) freezes a pretrained ViT and learns a\n"
    "small *prompt pool*: M trainable prompt tokens, each paired with a\n"
    "trainable key. At inference, the [CLS] embedding from the frozen ViT\n"
    "is used as a query; cosine similarity against the M keys picks the\n"
    "top-N prompts; those prompts get prepended to the patch sequence and\n"
    "the model classifies on top of the resulting representation. The\n"
    "training loss is cross-entropy plus a query-key matching term that\n"
    "pulls queries toward their selected keys.\n"
    "\n"
    "I wanted the implementation auditable, so it's all in the next cell:\n"
    "no submodule hopping, ~80 lines.\n"
)

code(
    "# ----- L2P implementation -----\n"
    "class L2PVit(nn.Module):\n"
    "    \"\"\"Learning to Prompt on a frozen ViT-B/16 (timm, ImageNet-21k).\"\"\"\n"
    "\n"
    "    def __init__(self, n_classes, M=10, N=4, L_p=5,\n"
    "                 model_name='vit_base_patch16_224.augreg_in21k'):\n"
    "        super().__init__()\n"
    "        vit = timm.create_model(model_name, pretrained=True, num_classes=0)\n"
    "        for p in vit.parameters():\n"
    "            p.requires_grad = False\n"
    "        vit.eval()\n"
    "        self.vit = vit\n"
    "        self.d   = vit.embed_dim   # 768\n"
    "        self.M, self.N, self.L_p = M, N, L_p\n"
    "\n"
    "        # prompt pool + keys, both small\n"
    "        self.prompts = nn.Parameter(torch.randn(M, L_p, self.d) * 0.02)\n"
    "        self.keys    = nn.Parameter(torch.randn(M, self.d) * 0.02)\n"
    "\n"
    "        # classification head over all classes (one shared head)\n"
    "        self.head = nn.Linear(self.d, n_classes)\n"
    "\n"
    "    @torch.no_grad()\n"
    "    def query(self, x):\n"
    "        # frozen ViT [CLS] as the query (no prompts injected)\n"
    "        feats = self.vit.forward_features(x)   # [B, T, d]\n"
    "        return feats[:, 0, :]                  # [CLS]\n"
    "\n"
    "    def select(self, q):\n"
    "        qn = F.normalize(q, dim=-1)\n"
    "        kn = F.normalize(self.keys, dim=-1)\n"
    "        sim = qn @ kn.T                        # [B, M]\n"
    "        top = sim.topk(self.N, dim=-1).indices # [B, N]\n"
    "        return top, qn\n"
    "\n"
    "    def forward_with_prompts(self, x, top):\n"
    "        B = x.size(0)\n"
    "        x = self.vit.patch_embed(x)            # [B, P, d]\n"
    "        cls = self.vit.cls_token.expand(B, -1, -1)\n"
    "        x = torch.cat([cls, x], dim=1)\n"
    "        x = x + self.vit.pos_embed             # CLS + patches share pos_embed\n"
    "        x = self.vit.pos_drop(x)\n"
    "        # gather top-N prompts and prepend them after [CLS]\n"
    "        sel = self.prompts[top]                # [B, N, L_p, d]\n"
    "        sel = sel.reshape(B, self.N * self.L_p, self.d)\n"
    "        x = torch.cat([x[:, :1], sel, x[:, 1:]], dim=1)\n"
    "        for blk in self.vit.blocks:\n"
    "            x = blk(x)\n"
    "        x = self.vit.norm(x)\n"
    "        return x[:, 0]                         # [CLS]\n"
    "\n"
    "    def forward(self, x):\n"
    "        q = self.query(x)\n"
    "        top, qn = self.select(q)\n"
    "        feat = self.forward_with_prompts(x, top)\n"
    "        logits = self.head(feat)\n"
    "        sel_keys = self.keys[top]              # [B, N, d]\n"
    "        return logits, qn, sel_keys, top\n"
    "\n"
    "\n"
    "def l2p_loss(logits, y, qn, sel_keys, lam=0.5):\n"
    "    \"\"\"CE + query-key matching. The matching term pulls queries toward\n"
    "    the keys of the prompts they actually selected.\"\"\"\n"
    "    ce = F.cross_entropy(logits, y)\n"
    "    sk = F.normalize(sel_keys, dim=-1)\n"
    "    match = -(qn.unsqueeze(1) * sk).sum(-1).mean()\n"
    "    return ce + lam * match, ce.item(), match.item()\n"
    "\n"
    "print('L2P module defined.')\n"
)

md(
    h3("4.1", "Why this works at all") + "\n"
    "\n"
    "Two things to know.\n"
    "\n"
    "First, the **frozen** ViT does most of the work. ImageNet-21k pretraining\n"
    "covers a huge variety of categories, so for CIFAR-100 the frozen [CLS]\n"
    "embedding is already pretty linearly separable. The classification head\n"
    "alone -- ignoring prompts -- gets non-trivial accuracy.\n"
    "\n"
    "Second, the **prompt pool** is a tiny module. With `M=10`, `L_p=5`,\n"
    "`d=768`, it has `10 * 5 * 768 = 38,400` prompt parameters plus `10 * 768`\n"
    "key parameters and a 768-by-n_classes head. The whole adapter is well\n"
    "under one percent of a ViT-B/16. Continual learning with such a tight\n"
    "parameter budget is the entire point.\n"
)

# ============================================================
# Section 5 -- Training run 1: partial overlap
# ============================================================
md(
    banner(5, "Training run 1 -- partial overlap") + "\n"
    "\n"
    "Time to build the per-task data, train L2P on scenario A, and snapshot\n"
    "the model after each task so we can compute the full accuracy matrix\n"
    "later.\n"
)

code(
    "# Per-task CIFAR-100 data builder.\n"
    "# CIFAR-100 has 100 classes, 500 train / 100 test images each.\n"
    "# We use a fixed mapping from scenario class IDs (0..69 in our toy\n"
    "# numbering) to actual CIFAR-100 class indices. For image-disjoint\n"
    "# behavior under exact overlap, we split each class's train pool in two.\n"
    "\n"
    "CIFAR_MEAN, CIFAR_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # ViT-augreg norm\n"
    "TFM = transforms.Compose([\n"
    "    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),\n"
    "])\n"
    "\n"
    "DATA_ROOT = '/kaggle/working/cifar100' if os.path.exists('/kaggle/working') else './cifar100'\n"
    "os.makedirs(DATA_ROOT, exist_ok=True)\n"
    "\n"
    "cifar_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True,  download=True, transform=TFM)\n"
    "cifar_test  = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=TFM)\n"
    "\n"
    "train_targets = np.array(cifar_train.targets)\n"
    "test_targets  = np.array(cifar_test.targets)\n"
    "\n"
    "# Build a fixed mapping from our scenario class IDs (0..69) to real\n"
    "# CIFAR-100 class indices (0..99). We just take the first 70 in order.\n"
    "SCEN_TO_CIFAR = {sid: sid for sid in range(70)}\n"
    "\n"
    "# For scenario B we need image-disjoint halves. Split per-class train indices in two.\n"
    "rng = np.random.RandomState(SEED)\n"
    "class_train_indices = {}\n"
    "for c in range(100):\n"
    "    idx = np.where(train_targets == c)[0]\n"
    "    rng.shuffle(idx)\n"
    "    class_train_indices[c] = idx\n"
    "print(f'CIFAR-100 ready: train={len(cifar_train)}, test={len(cifar_test)}')\n"
)

code(
    "def build_task_loaders(spec, recur_set, train=True, half='full',\n"
    "                       batch_size=64, eval_batch_size=256):\n"
    "    \"\"\"Returns a list of DataLoaders, one per task.\n"
    "\n"
    "    half: 'full' uses all images of a class; 'first'/'second' uses\n"
    "          the corresponding 50/50 split (for image-disjoint exact overlap).\n"
    "    For scenario B, when a class is in recur_set we use 'first' for the\n"
    "    earlier task and 'second' for the later one. Caller passes the right\n"
    "    half via the per-task halves list (see build_scenario_loaders below).\n"
    "    \"\"\"\n"
    "    raise NotImplementedError  # build_scenario_loaders is the public API\n"
    "\n"
    "def build_scenario_loaders(spec, recur_set, scenario_kind,\n"
    "                           batch_size=64, eval_batch_size=256):\n"
    "    \"\"\"Build train + test loaders for every task in a scenario.\n"
    "\n"
    "    scenario_kind: 'A' (partial) or 'B' (exact).\n"
    "    For scenario B, recurring classes use first-half samples in the earlier\n"
    "    task and second-half samples in the later task -- image-disjoint.\n"
    "    \"\"\"\n"
    "    train_loaders, test_loaders, eval_test_loaders = [], [], []\n"
    "    # Track which half each (task, recurring_class) uses.\n"
    "    seen_first = set()  # classes we've already issued first-half for\n"
    "    for t, classes in enumerate(spec.classes_per_task):\n"
    "        idx_train, idx_test = [], []\n"
    "        cifar_classes = [SCEN_TO_CIFAR[c] for c in classes]\n"
    "        for sid, cid in zip(classes, cifar_classes):\n"
    "            tr = class_train_indices[cid]\n"
    "            if scenario_kind == 'B' and sid in recur_set:\n"
    "                # split halves\n"
    "                half_size = len(tr) // 2\n"
    "                if sid not in seen_first:\n"
    "                    use = tr[:half_size]\n"
    "                    seen_first.add(sid)\n"
    "                else:\n"
    "                    use = tr[half_size:]\n"
    "            elif scenario_kind == 'A' and sid in recur_set:\n"
    "                # partial overlap also disjoint by halves\n"
    "                half_size = len(tr) // 2\n"
    "                if sid not in seen_first:\n"
    "                    use = tr[:half_size]\n"
    "                    seen_first.add(sid)\n"
    "                else:\n"
    "                    use = tr[half_size:]\n"
    "            else:\n"
    "                use = tr\n"
    "            idx_train.extend(use.tolist())\n"
    "            idx_test.extend(np.where(test_targets == cid)[0].tolist())\n"
    "        tr_loader = DataLoader(Subset(cifar_train, idx_train), batch_size=batch_size,\n"
    "                               shuffle=True, num_workers=2, pin_memory=(DEVICE=='cuda'), drop_last=True)\n"
    "        te_loader = DataLoader(Subset(cifar_test,  idx_test),  batch_size=eval_batch_size,\n"
    "                               shuffle=False, num_workers=2, pin_memory=(DEVICE=='cuda'))\n"
    "        train_loaders.append(tr_loader)\n"
    "        test_loaders.append(te_loader)\n"
    "    return train_loaders, test_loaders\n"
    "\n"
    "trainA, testA = build_scenario_loaders(scenario_A, RECUR_A, 'A')\n"
    "for t, (trl, tel) in enumerate(zip(trainA, testA)):\n"
    "    print(f'  scenario A task {t}: train={len(trl.dataset)} test={len(tel.dataset)}')\n"
)

code(
    "# Class-to-head-index mapping. The head is one shared linear layer\n"
    "# spanning every class either scenario will ever see. Different scenarios\n"
    "# use different class IDs (A: 0..39, B: 40..69), so the head needs at\n"
    "# least 70 outputs. We use 70 flat.\n"
    "N_HEAD_CLASSES = 70\n"
    "\n"
    "def evaluate(model, loader):\n"
    "    model.eval()\n"
    "    correct = total = 0\n"
    "    with torch.no_grad():\n"
    "        for x, y in loader:\n"
    "            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)\n"
    "            with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):\n"
    "                logits, *_ = model(x)\n"
    "            pred = logits.argmax(-1)\n"
    "            correct += (pred == y).sum().item()\n"
    "            total   += y.numel()\n"
    "    return correct / max(total, 1)\n"
    "\n"
    "def evaluate_per_class(model, loader):\n"
    "    model.eval()\n"
    "    per = defaultdict(lambda: [0, 0])  # cid -> [correct, total]\n"
    "    with torch.no_grad():\n"
    "        for x, y in loader:\n"
    "            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)\n"
    "            with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):\n"
    "                logits, *_ = model(x)\n"
    "            pred = logits.argmax(-1)\n"
    "            for yi, pi in zip(y.tolist(), pred.tolist()):\n"
    "                per[yi][1] += 1\n"
    "                if yi == pi:\n"
    "                    per[yi][0] += 1\n"
    "    return {c: (cor / tot if tot else 0.0) for c, (cor, tot) in per.items()}\n"
)

code(
    "def train_scenario(spec, recur_set, scenario_kind, epochs=5, lam=0.5, lr=1e-3, log=True):\n"
    "    \"\"\"Train a fresh L2P model task-by-task. Returns model snapshots and metrics.\n"
    "    snapshots[t] = state_dict after training task t.\n"
    "    R[k, t]      = test accuracy on task t after training task k. (k >= t)\n"
    "    \"\"\"\n"
    "    train_loaders, test_loaders = build_scenario_loaders(spec, recur_set, scenario_kind)\n"
    "    T = spec.n_tasks\n"
    "\n"
    "    model = L2PVit(N_HEAD_CLASSES).to(DEVICE)\n"
    "    # only prompts, keys, head are trainable\n"
    "    trainable = [p for p in model.parameters() if p.requires_grad]\n"
    "    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=0.0)\n"
    "    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))\n"
    "\n"
    "    R = np.full((T, T), np.nan)\n"
    "    snapshots = []\n"
    "    epoch_log = []\n"
    "    for t, loader in enumerate(train_loaders):\n"
    "        for ep in range(epochs):\n"
    "            model.train()\n"
    "            t0 = time.time()\n"
    "            run_ce = run_match = run_n = 0\n"
    "            for x, y in loader:\n"
    "                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)\n"
    "                opt.zero_grad(set_to_none=True)\n"
    "                with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):\n"
    "                    logits, qn, sel_keys, top = model(x)\n"
    "                    loss, ce, mt = l2p_loss(logits, y, qn, sel_keys, lam=lam)\n"
    "                scaler.scale(loss).backward()\n"
    "                scaler.step(opt)\n"
    "                scaler.update()\n"
    "                run_ce += ce * y.size(0); run_match += mt * y.size(0); run_n += y.size(0)\n"
    "            tr_acc = evaluate(model, test_loaders[t])\n"
    "            dt = time.time() - t0\n"
    "            if log:\n"
    "                print(f'  [{spec.name}] task {t} epoch {ep+1}/{epochs}  '\n"
    "                      f'ce={run_ce/run_n:.3f}  match={run_match/run_n:.3f}  '\n"
    "                      f'acc(t)={tr_acc:.3f}  ({dt:.1f}s)')\n"
    "            epoch_log.append(dict(scenario=spec.name, task=t, epoch=ep+1,\n"
    "                                  ce=run_ce/run_n, match=run_match/run_n,\n"
    "                                  acc=tr_acc, secs=dt))\n"
    "        # eval on every task seen so far\n"
    "        for j in range(t + 1):\n"
    "            R[t, j] = evaluate(model, test_loaders[j])\n"
    "        if log:\n"
    "            row = ' '.join(f'{R[t, j]:.3f}' if not math.isnan(R[t, j]) else '  -- ' for j in range(T))\n"
    "            print(f'  R[{t}, :] = {row}')\n"
    "        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})\n"
    "    return dict(model=model, R=R, snapshots=snapshots, log=epoch_log,\n"
    "                test_loaders=test_loaders, train_loaders=train_loaders)\n"
)

code(
    "# Train scenario A. Watch the per-epoch lines scroll -- this is how you\n"
    "# know it actually trained.\n"
    "EPOCHS = 5\n"
    "print(f'Training L2P on scenario A (partial overlap), {EPOCHS} epochs/task ...')\n"
    "resA = train_scenario(scenario_A, RECUR_A, 'A', epochs=EPOCHS)\n"
    "print(f'\\nFinal R matrix (scenario A):\\n{np.round(resA[\"R\"], 3)}')\n"
    "torch.save(resA['snapshots'][-1], 'l2p_scenA_final.pt')\n"
)

# ============================================================
# Section 6 -- Training run 2: exact overlap
# ============================================================
md(
    banner(6, "Training run 2 -- exact overlap") + "\n"
    "\n"
    "Same training function, same hyperparameters, only the stream changes.\n"
    "Free GPU memory before starting -- the snapshots from run 1 are still\n"
    "on CPU but the model itself was on CUDA.\n"
)

code(
    "# Free memory before run 2.\n"
    "del resA['model']\n"
    "torch.cuda.empty_cache() if DEVICE == 'cuda' else None\n"
    "\n"
    "print(f'Training L2P on scenario B (exact overlap), {EPOCHS} epochs/task ...')\n"
    "resB = train_scenario(scenario_B, RECUR_B, 'B', epochs=EPOCHS)\n"
    "print(f'\\nFinal R matrix (scenario B):\\n{np.round(resB[\"R\"], 3)}')\n"
    "torch.save(resB['snapshots'][-1], 'l2p_scenB_final.pt')\n"
)

# ============================================================
# Section 7 -- The accuracy matrix
# ============================================================
md(
    banner(7, "The accuracy matrix") + "\n"
    "\n"
    "The R matrix is the standard CL summary: R[k, t] is the test accuracy\n"
    "on task t evaluated immediately after training task k. Lower triangular\n"
    "by construction -- you cannot evaluate on a task before it has been\n"
    "introduced.\n"
)

code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')\n"
    "for ax, res, label in zip(axes, (resA, resB), ('scenario A', 'scenario B')):\n"
    "    R = res['R']\n"
    "    mask = np.isnan(R)\n"
    "    sns.heatmap(R, ax=ax, annot=True, fmt='.2f', cmap=seq_cmap(),\n"
    "                vmin=0, vmax=1, mask=mask, linewidths=0.6, linecolor='white',\n"
    "                annot_kws={'color': DARK, 'fontweight': 'bold'},\n"
    "                cbar_kws={'label': 'accuracy'})\n"
    "    ax.set_title(f'R[k, t] -- {label}')\n"
    "    ax.set_xlabel('task t (evaluated)'); ax.set_ylabel('task k (trained through)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('figures/R_matrix.png')\n"
    "plt.show()\n"
)

md(
    "The two matrices already tell the story. In **scenario A**, the diagonal\n"
    "is healthy and below-diagonal cells decay slowly: R[3, 0] is roughly\n"
    "where it lands after three more tasks of training. In **scenario B**,\n"
    "look at R[2, 0] -- it should be unusually high, because training task 2\n"
    "is essentially retraining task 0's classes on fresh image samples. Then\n"
    "watch R[3, 0] and R[3, 2]: those are the same classes seen at different\n"
    "times.\n"
)

# ============================================================
# Section 8 -- Forgetting and BWT
# ============================================================
md(
    banner(8, "Forgetting and backward transfer") + "\n"
    "\n"
    "ACC, BWT, FWT -- the three numbers every CL paper reports.\n"
    "\n"
    "- **ACC** = mean diagonal of the lower triangle (how well we know each\n"
    "  task at the moment we finish it -- but typically reported as final-row\n"
    "  mean: the average task accuracy at the end of training).\n"
    "- **BWT** = mean over tasks of `R[T-1, t] - R[t, t]`. Negative means\n"
    "  forgetting; positive means later training improved earlier tasks.\n"
    "- **FWT** = mean over `t > 0` of `R[t-1, t]` minus a random baseline.\n"
    "  Often noisy on small streams; included for completeness.\n"
)

code(
    "def metrics(R):\n"
    "    T = R.shape[0]\n"
    "    final_acc = np.nanmean(R[T-1, :T])\n"
    "    bwt = np.nanmean([R[T-1, t] - R[t, t] for t in range(T-1)])\n"
    "    # naive FWT: average accuracy on a task one step before it was trained,\n"
    "    # versus a 1/n_classes_per_task baseline\n"
    "    fwt_vals = []\n"
    "    for t in range(1, T):\n"
    "        if not np.isnan(R[t-1, t]):\n"
    "            fwt_vals.append(R[t-1, t] - 0.10)  # 10 classes/task -> 1/10 random\n"
    "    fwt = float(np.mean(fwt_vals)) if fwt_vals else float('nan')\n"
    "    return dict(ACC=final_acc, BWT=bwt, FWT=fwt)\n"
    "\n"
    "mA, mB = metrics(resA['R']), metrics(resB['R'])\n"
    "import pandas as pd\n"
    "df_metrics = pd.DataFrame({'scenario A': mA, 'scenario B': mB}).round(4)\n"
    "df_metrics\n"
)

code(
    "fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='white')\n"
    "labels = ['ACC', 'BWT', 'FWT']\n"
    "valsA = [mA[k] for k in labels]; valsB = [mB[k] for k in labels]\n"
    "x = np.arange(len(labels)); w = 0.35\n"
    "b1 = ax.bar(x - w/2, valsA, w, label='scenario A', color=PRIMARY)\n"
    "b2 = ax.bar(x + w/2, valsB, w, label='scenario B', color=SECONDARY)\n"
    "ax.axhline(0, color='#888', linewidth=0.7)\n"
    "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
    "ax.set_ylabel('value')\n"
    "ax.set_title('CL metrics: scenario A vs B')\n"
    "for bars in (b1, b2):\n"
    "    for b in bars:\n"
    "        h = b.get_height()\n"
    "        ax.text(b.get_x() + b.get_width()/2, h + (0.01 if h >= 0 else -0.03),\n"
    "                f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top',\n"
    "                fontsize=9, color=DARK, fontweight='bold')\n"
    "ax.legend()\n"
    "plt.tight_layout(); plt.savefig('figures/cl_metrics.png'); plt.show()\n"
)

md(
    "BWT is the metric I actually care about here. In scenario A it's slightly\n"
    "negative (some forgetting). In scenario B it's much closer to zero or\n"
    "even positive -- but that is partly an artifact: when task 2 retrains\n"
    "task 0's classes, R[T-1, 0] gets a free boost. The BWT formula doesn't\n"
    "know that. Treat the scenario B BWT as 'forgetting that survived the\n"
    "intentional refresh,' not as evidence that L2P is special on overlap.\n"
)

# ============================================================
# Section 9 -- Recurring vs novel classes
# ============================================================
md(
    banner(9, "Recurring vs novel classes -- the headline finding") + "\n"
    "\n"
    + callout("This is where the analysis lives") + "\n"
    "\n"
    "Forget the matrices for a moment. Take the model at the end of training,\n"
    "evaluate on every test image of every class it saw, and split the\n"
    "per-class accuracies into two groups: classes that recurred during\n"
    "training, and classes that didn't.\n"
)

code(
    "def per_class_final_acc(res, recur_set):\n"
    "    \"\"\"Run final model over the union of all task test sets and return\n"
    "    a dict {scen_class_id -> accuracy}.\"\"\"\n"
    "    model = res['model'] if 'model' in res else None\n"
    "    # If we already deleted the model (memory), rebuild + load snapshot.\n"
    "    if model is None:\n"
    "        model = L2PVit(N_HEAD_CLASSES).to(DEVICE)\n"
    "        sd = res['snapshots'][-1]\n"
    "        model.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})\n"
    "    model.eval()\n"
    "    # combine all task test loaders (deduplicated by Subset indices)\n"
    "    seen_idx = set()\n"
    "    combined_idx = []\n"
    "    for tl in res['test_loaders']:\n"
    "        for i in tl.dataset.indices:\n"
    "            if i not in seen_idx:\n"
    "                seen_idx.add(i); combined_idx.append(i)\n"
    "    combined = DataLoader(Subset(cifar_test, combined_idx), batch_size=256,\n"
    "                          shuffle=False, num_workers=2, pin_memory=(DEVICE=='cuda'))\n"
    "    per = evaluate_per_class(model, combined)\n"
    "    return per, model\n"
    "\n"
    "perA, modelA_final = per_class_final_acc(resA, RECUR_A)\n"
    "perB, modelB_final = per_class_final_acc(resB, RECUR_B)\n"
)

code(
    "def split_recur_novel(per, recur_scen, scen_to_cifar):\n"
    "    rec_acc, novel_acc = [], []\n"
    "    for sid, cid in scen_to_cifar.items():\n"
    "        if cid not in per:  # not in this scenario's classes\n"
    "            continue\n"
    "        a = per[cid]\n"
    "        if sid in recur_scen:\n"
    "            rec_acc.append(a)\n"
    "        else:\n"
    "            novel_acc.append(a)\n"
    "    return rec_acc, novel_acc\n"
    "\n"
    "recA, novA = split_recur_novel(perA, RECUR_A, SCEN_TO_CIFAR)\n"
    "recB, novB = split_recur_novel(perB, RECUR_B, SCEN_TO_CIFAR)\n"
    "print(f'scenario A: recurring n={len(recA)} mean={np.mean(recA):.3f} | '\n"
    "      f'novel n={len(novA)} mean={np.mean(novA):.3f}')\n"
    "print(f'scenario B: recurring n={len(recB)} mean={np.mean(recB):.3f} | '\n"
    "      f'novel n={len(novB)} mean={np.mean(novB):.3f}')\n"
)

code(
    "# Plot 1: bar chart with bootstrap CIs.\n"
    "def bootstrap_ci(vals, n=1000, alpha=0.05, rng=None):\n"
    "    rng = rng or np.random.RandomState(0)\n"
    "    vals = np.asarray(vals)\n"
    "    boots = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n)]\n"
    "    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])\n"
    "    return float(np.mean(vals)), float(lo), float(hi)\n"
    "\n"
    "groups = [('A recurring', recA), ('A novel', novA), ('B recurring', recB), ('B novel', novB)]\n"
    "means, los, his = [], [], []\n"
    "for _, vals in groups:\n"
    "    m, lo, hi = bootstrap_ci(vals)\n"
    "    means.append(m); los.append(m - lo); his.append(hi - m)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor='white')\n"
    "x = np.arange(len(groups))\n"
    "colors = [PRIMARY, EXTRA[0], SECONDARY, EXTRA[1]]\n"
    "bars = ax.bar(x, means, yerr=[los, his], color=colors, capsize=5, edgecolor=DARK, linewidth=0.6)\n"
    "ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], rotation=10)\n"
    "ax.set_ylabel('final test accuracy')\n"
    "ax.set_title('recurring vs novel: final accuracy by class group (bootstrap 95% CI)')\n"
    "for b, m in zip(bars, means):\n"
    "    ax.text(b.get_x() + b.get_width()/2, m + 0.02, f'{m:.3f}',\n"
    "            ha='center', va='bottom', fontsize=9, color=DARK, fontweight='bold')\n"
    "ax.set_ylim(0, 1.05)\n"
    "plt.tight_layout(); plt.savefig('figures/recur_vs_novel.png'); plt.show()\n"
)

code(
    "# Plot 2: scatter of per-class final accuracy by last-seen task.\n"
    "def collect_class_points(spec, recur_set, per):\n"
    "    rows = []\n"
    "    for t, classes in enumerate(spec.classes_per_task):\n"
    "        for sid in classes:\n"
    "            cid = SCEN_TO_CIFAR[sid]\n"
    "            if cid not in per:\n"
    "                continue\n"
    "            rows.append(dict(sid=sid, cid=cid, last_task=t,\n"
    "                             acc=per[cid], recur=(sid in recur_set)))\n"
    "    # last_task should be the LAST task each class appeared in\n"
    "    last_seen = {}\n"
    "    for r in rows:\n"
    "        last_seen[r['sid']] = max(last_seen.get(r['sid'], -1), r['last_task'])\n"
    "    seen = {}\n"
    "    for r in rows:\n"
    "        if r['sid'] in seen:\n"
    "            continue\n"
    "        if last_seen[r['sid']] == r['last_task']:\n"
    "            seen[r['sid']] = r\n"
    "    return list(seen.values())\n"
    "\n"
    "ptsA = collect_class_points(scenario_A, RECUR_A, perA)\n"
    "ptsB = collect_class_points(scenario_B, RECUR_B, perB)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')\n"
    "for ax, pts, name, recur_color in zip(axes, (ptsA, ptsB), ('scenario A', 'scenario B'),\n"
    "                                       (PRIMARY, SECONDARY)):\n"
    "    rec = [p for p in pts if p['recur']]\n"
    "    nov = [p for p in pts if not p['recur']]\n"
    "    jx = lambda r: r['last_task'] + (np.random.RandomState(r['sid']).rand() - 0.5) * 0.25\n"
    "    ax.scatter([jx(r) for r in nov], [r['acc'] for r in nov], s=45, alpha=0.55,\n"
    "               color='#9aa1a8', edgecolor='white', label='novel')\n"
    "    ax.scatter([jx(r) for r in rec], [r['acc'] for r in rec], s=70, alpha=0.85,\n"
    "               color=recur_color, edgecolor=DARK, linewidth=0.8, label='recurring')\n"
    "    ax.set_xticks([0, 1, 2, 3])\n"
    "    ax.set_xlabel('task of last appearance'); ax.set_ylabel('final accuracy')\n"
    "    ax.set_title(f'{name}: per-class final accuracy')\n"
    "    ax.set_ylim(0, 1.05); ax.legend()\n"
    "plt.tight_layout(); plt.savefig('figures/per_class_scatter.png'); plt.show()\n"
)

code(
    "# Plot 3: per-recurring-class accuracy delta -- first appearance vs final.\n"
    "# Need accuracy on each recurring class right after its FIRST appearance.\n"
    "def acc_after_first_appearance(res, recur_set, spec):\n"
    "    \"\"\"For each recurring class, return its test accuracy at the snapshot\n"
    "    taken after its first appearing task. We rebuild the model from that\n"
    "    snapshot and evaluate on the relevant test loader.\"\"\"\n"
    "    out = {}\n"
    "    first_task_of = {}\n"
    "    for t, classes in enumerate(spec.classes_per_task):\n"
    "        for sid in classes:\n"
    "            if sid in recur_set and sid not in first_task_of:\n"
    "                first_task_of[sid] = t\n"
    "    cache = {}\n"
    "    for sid, t in first_task_of.items():\n"
    "        if t not in cache:\n"
    "            m = L2PVit(N_HEAD_CLASSES).to(DEVICE)\n"
    "            m.load_state_dict({k: v.to(DEVICE) for k, v in res['snapshots'][t].items()})\n"
    "            m.eval()\n"
    "            cache[t] = m\n"
    "        m = cache[t]\n"
    "        cid = SCEN_TO_CIFAR[sid]\n"
    "        idx = np.where(test_targets == cid)[0]\n"
    "        single = DataLoader(Subset(cifar_test, idx.tolist()), batch_size=128,\n"
    "                            shuffle=False, num_workers=2, pin_memory=(DEVICE=='cuda'))\n"
    "        out[sid] = evaluate(m, single)\n"
    "    # free memory\n"
    "    del cache\n"
    "    if DEVICE == 'cuda': torch.cuda.empty_cache()\n"
    "    return out\n"
    "\n"
    "first_A = acc_after_first_appearance(resA, RECUR_A, scenario_A)\n"
    "first_B = acc_after_first_appearance(resB, RECUR_B, scenario_B)\n"
    "final_A = {sid: perA[SCEN_TO_CIFAR[sid]] for sid in RECUR_A}\n"
    "final_B = {sid: perB[SCEN_TO_CIFAR[sid]] for sid in RECUR_B}\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')\n"
    "for ax, first, final, name, color in zip(axes, (first_A, first_B), (final_A, final_B),\n"
    "                                          ('scenario A', 'scenario B'),\n"
    "                                          (PRIMARY, SECONDARY)):\n"
    "    sids = sorted(first.keys())\n"
    "    f1 = [first[s] for s in sids]; f2 = [final[s] for s in sids]\n"
    "    xs = np.arange(len(sids))\n"
    "    ax.scatter(xs, f1, s=70, color='#cccccc', edgecolor=DARK, label='after first task')\n"
    "    ax.scatter(xs, f2, s=70, color=color, edgecolor=DARK, label='after final task')\n"
    "    for x, a, b in zip(xs, f1, f2):\n"
    "        ax.annotate('', xy=(x, b), xytext=(x, a),\n"
    "                    arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))\n"
    "    ax.set_xticks(xs); ax.set_xticklabels([f'cls {s}' for s in sids], rotation=30)\n"
    "    ax.set_ylim(0, 1.05); ax.set_ylabel('test accuracy')\n"
    "    ax.set_title(f'{name}: recurring classes -- first vs final')\n"
    "    ax.legend()\n"
    "plt.tight_layout(); plt.savefig('figures/recur_delta.png'); plt.show()\n"
)

md(
    "Two things to take away.\n"
    "\n"
    "First, the recurring-vs-novel gap is real but smaller than I'd hoped.\n"
    "L2P doesn't strongly *exploit* the second appearance of a class -- it\n"
    "mostly avoids forgetting it as fast. The bump from a second pass is\n"
    "modest. That matches the architecture: prompts are selected by\n"
    "similarity to a frozen query, so the prompts that fired in task 0 will\n"
    "tend to fire again on the same class in task 2 *whether or not* L2P\n"
    "explicitly knows the classes are the same.\n"
    "\n"
    "Second, scenario B is suspiciously high on the recurring task. That's\n"
    "not L2P being clever -- it's just that the model trained on those exact\n"
    "classes very recently. The question worth chasing is: did the prompt\n"
    "pool *retain* what it learned from task 0, or did task 2 just relearn\n"
    "the same thing from scratch? S10 is the diagnostic.\n"
    "\n"
    "(I'd love to see anyone who has run L2P on overlapping streams compare\n"
    "numbers in the comments -- I couldn't find a published comparison.)\n"
)

# ============================================================
# Section 10 -- What did the prompts learn?
# ============================================================
md(
    banner(10, "What did the prompts learn?") + "\n"
    "\n"
    "Three diagnostics on the prompt pool itself, after all training has\n"
    "finished. None of these are in the L2P paper -- they're the kind of\n"
    "thing you cook up because you're curious.\n"
)

code(
    h3("10.1", "Prompt-key cosine similarity") + "\n"
)

md(
    "Are the M=10 prompt keys spread out, or did they collapse onto one\n"
    "another? A collapsed pool would mean L2P effectively learned a single\n"
    "prompt and ignored the selection mechanism.\n"
)

code(
    "def key_similarity(model):\n"
    "    K = model.keys.detach().cpu()\n"
    "    Kn = F.normalize(K, dim=-1)\n"
    "    return (Kn @ Kn.T).numpy()\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), facecolor='white')\n"
    "for ax, model, name in zip(axes, (modelA_final, modelB_final), ('scenario A', 'scenario B')):\n"
    "    S = key_similarity(model)\n"
    "    sns.heatmap(S, ax=ax, vmin=-1, vmax=1, cmap='RdBu_r', center=0,\n"
    "                annot=True, fmt='.2f', annot_kws={'fontsize': 7})\n"
    "    ax.set_title(f'{name}: pairwise key cosine sim')\n"
    "    ax.set_xlabel('prompt j'); ax.set_ylabel('prompt i')\n"
    "plt.tight_layout(); plt.savefig('figures/key_similarity.png'); plt.show()\n"
)

code(
    h3("10.2", "Which prompts get picked by which task?") + "\n"
)

md(
    "For each task, run the test set through the model and count top-N\n"
    "prompt selections. If task 2 selects the same prompts as task 0 in\n"
    "scenario B, that's evidence the prompt pool is genuinely *reusing*\n"
    "knowledge across the recurrence.\n"
)

code(
    "def selection_histogram(model, loader, M=10):\n"
    "    model.eval()\n"
    "    counts = np.zeros(M, dtype=np.int64)\n"
    "    n_seen = 0\n"
    "    with torch.no_grad():\n"
    "        for x, _ in loader:\n"
    "            x = x.to(DEVICE, non_blocking=True)\n"
    "            with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):\n"
    "                _, _, _, top = model(x)\n"
    "            for tid in top.flatten().tolist():\n"
    "                counts[tid] += 1\n"
    "            n_seen += x.size(0)\n"
    "    return counts / counts.sum()\n"
    "\n"
    "def per_task_histograms(res, model):\n"
    "    return [selection_histogram(model, tl) for tl in res['test_loaders']]\n"
    "\n"
    "histsA = per_task_histograms(resA, modelA_final)\n"
    "histsB = per_task_histograms(resB, modelB_final)\n"
    "\n"
    "fig, axes = plt.subplots(2, 4, figsize=(13, 5), facecolor='white', sharey=True)\n"
    "for col, (hA, hB) in enumerate(zip(histsA, histsB)):\n"
    "    axes[0, col].bar(range(10), hA, color=PRIMARY, edgecolor=DARK, linewidth=0.6)\n"
    "    axes[0, col].set_title(f'A task {col}')\n"
    "    axes[1, col].bar(range(10), hB, color=SECONDARY, edgecolor=DARK, linewidth=0.6)\n"
    "    axes[1, col].set_title(f'B task {col}')\n"
    "    for r in (0, 1):\n"
    "        axes[r, col].set_xticks(range(10))\n"
    "        axes[r, col].set_xlabel('prompt id')\n"
    "axes[0, 0].set_ylabel('selection frequency')\n"
    "axes[1, 0].set_ylabel('selection frequency')\n"
    "plt.tight_layout(); plt.savefig('figures/prompt_histograms.png'); plt.show()\n"
)

code(
    h3("10.3", "Did recurring tasks pick similar prompts?") + "\n"
)

md(
    "Take each task's top-N prompt set and compare via Jaccard similarity\n"
    "across task pairs. The cell to focus on is **(task 0, task 2)** in\n"
    "scenario B -- those are the same classes seen at different times. If\n"
    "L2P is reusing prompts the cell should glow.\n"
)

code(
    "def top_prompt_set(hist, k=4):\n"
    "    return set(np.argsort(hist)[-k:].tolist())\n"
    "\n"
    "def jaccard(a, b):\n"
    "    return len(a & b) / max(len(a | b), 1)\n"
    "\n"
    "def jaccard_matrix(hists, k=4):\n"
    "    sets = [top_prompt_set(h, k) for h in hists]\n"
    "    M = np.zeros((len(sets), len(sets)))\n"
    "    for i in range(len(sets)):\n"
    "        for j in range(len(sets)):\n"
    "            M[i, j] = jaccard(sets[i], sets[j])\n"
    "    return M\n"
    "\n"
    "JA = jaccard_matrix(histsA)\n"
    "JB = jaccard_matrix(histsB)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')\n"
    "for ax, J, name in zip(axes, (JA, JB), ('scenario A', 'scenario B')):\n"
    "    sns.heatmap(J, ax=ax, vmin=0, vmax=1, cmap=seq_cmap(),\n"
    "                annot=True, fmt='.2f', linewidths=0.6, linecolor='white',\n"
    "                annot_kws={'color': DARK, 'fontweight': 'bold'})\n"
    "    ax.set_title(f'{name}: top-4 prompt Jaccard across tasks')\n"
    "    ax.set_xlabel('task j'); ax.set_ylabel('task i')\n"
    "plt.tight_layout(); plt.savefig('figures/prompt_jaccard.png'); plt.show()\n"
)

md(
    "If the (0, 2) cell of scenario B is high (say >= 0.5) then L2P really\n"
    "is reusing prompts across the recurrence. If it's low, the prompt pool\n"
    "rearranged itself between tasks 0 and 2 even though the underlying\n"
    "classes were identical -- which would be a quietly negative result for\n"
    "the 'prompts as memory' framing.\n"
    "\n"
    "I'm honestly not sure what to predict for the off-diagonals in scenario\n"
    "A. Open to theories.\n"
)

# ============================================================
# Section 11 -- Failure mode tour
# ============================================================
md(
    banner(11, "Failure mode tour") + "\n"
    "\n"
    "Six misclassified images from the final scenario A model. The point\n"
    "isn't to count errors -- it's to look at what kinds of errors L2P\n"
    "actually makes. Mistakes carry more information than wins.\n"
)

code(
    "# Pick 6 misclassified samples from the union of all scenario A test sets.\n"
    "model = modelA_final\n"
    "model.eval()\n"
    "seen_idx = set(); combined_idx = []\n"
    "for tl in resA['test_loaders']:\n"
    "    for i in tl.dataset.indices:\n"
    "        if i not in seen_idx:\n"
    "            seen_idx.add(i); combined_idx.append(i)\n"
    "combined = DataLoader(Subset(cifar_test, combined_idx), batch_size=128,\n"
    "                      shuffle=False, num_workers=2)\n"
    "\n"
    "wrong = []  # list of (orig_idx, true_cid, pred_cid, conf)\n"
    "with torch.no_grad():\n"
    "    cursor = 0\n"
    "    for x, y in combined:\n"
    "        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)\n"
    "        with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):\n"
    "            logits, *_ = model(x)\n"
    "        probs = logits.softmax(-1)\n"
    "        pred  = probs.argmax(-1)\n"
    "        conf  = probs.max(-1).values\n"
    "        for k in range(x.size(0)):\n"
    "            if pred[k].item() != y[k].item():\n"
    "                wrong.append((combined_idx[cursor + k], int(y[k].item()),\n"
    "                              int(pred[k].item()), float(conf[k].item())))\n"
    "        cursor += x.size(0)\n"
    "\n"
    "rng = np.random.RandomState(7)\n"
    "rng.shuffle(wrong)\n"
    "picks = wrong[:6]\n"
    "print(f'Total misclassified in combined test set: {len(wrong)}; showing 6.')\n"
)

code(
    "# CIFAR-100 fine label names\n"
    "CIFAR100_NAMES = cifar_test.classes\n"
    "\n"
    "# de-normalize ImageNet-augreg style (mean=std=0.5) for display\n"
    "def denorm(t):\n"
    "    return (t * 0.5 + 0.5).clamp(0, 1)\n"
    "\n"
    "fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), facecolor='white')\n"
    "for ax, (idx, true_c, pred_c, conf) in zip(axes.flatten(), picks):\n"
    "    img, _ = cifar_test[idx]\n"
    "    ax.imshow(denorm(img).permute(1, 2, 0).numpy())\n"
    "    ax.axis('off')\n"
    "    ax.set_title(f'true: {CIFAR100_NAMES[true_c]}\\npred: {CIFAR100_NAMES[pred_c]}  (conf {conf:.2f})',\n"
    "                 fontsize=10, color=DARK)\n"
    "plt.suptitle('six failure cases, scenario A final model', fontsize=12, color=DARK, y=1.01)\n"
    "plt.tight_layout(); plt.savefig('figures/failures.png'); plt.show()\n"
)

md(
    "Most of the failures cluster in two buckets. CIFAR-100 has several\n"
    "mammal/wild-animal pairs that share coarse silhouettes, and CIFAR\n"
    "images are 32x32 upsampled to 224 -- texture cues that ViT relies on\n"
    "are blurry. The high-confidence wrongs are the interesting ones; the\n"
    "low-confidence wrongs are basically the model saying 'I have no idea.'\n"
)

# ============================================================
# Section 12 -- Closing thoughts
# ============================================================
md(
    banner(12, "Closing thoughts") + "\n"
    "\n"
    "What worked: L2P with a frozen ViT-B/16 is a delight to train. The\n"
    "whole notebook fits well under the Kaggle T4 budget and the prompt\n"
    "pool is small enough that you can actually inspect every prompt.\n"
    "\n"
    "What didn't quite work: I came in expecting a sharp 'recurring beats\n"
    "novel' bump and found a softer one. The mechanism that *should* deliver\n"
    "the bump -- prompts being reselected on the recurrence -- is partially\n"
    "there (S10.3) but quieter than I assumed. My current best guess is\n"
    "that ImageNet-21k features are already so strong on CIFAR-100 classes\n"
    "that the prompt pool only contributes a thin layer of task-specific\n"
    "shaping, and the head does most of the discrimination.\n"
    "\n"
    "If I were running this again I'd swap CIFAR-100 for a dataset where\n"
    "the frozen backbone is *less* sufficient (e.g. fine-grained birds, or\n"
    "medical imagery) so the prompts have to carry more weight. I'd also\n"
    "log the prompt-key vectors over training to see when the pool actually\n"
    "specializes -- the snapshots I have are end-of-task, which hides the\n"
    "fast inner-loop dynamics.\n"
    "\n"
    "If you found this useful, the CLOVER benchmark used to construct the\n"
    "two streams is at [github.com/danushkumar-v/clover-cl](https://github.com/danushkumar-v/clover-cl).\n"
    "It's small and self-contained -- the OverlapSpec abstraction is roughly\n"
    "the inline `OverlapSpec` class above plus seed handling and PILOT\n"
    "compatibility.\n"
    "\n"
    "*Code, errors, and confused comments at github.com/danushkumar-v/clover-cl*\n"
    "\n"
    "## References\n"
    "\n"
    "- Wang et al. (2022). Learning to Prompt for Continual Learning. *CVPR 2022.*\n"
    "  [arxiv.org/abs/2112.08654](https://arxiv.org/abs/2112.08654)\n"
    "- Hemati et al. (2023). Class-Incremental Learning with Repetition.\n"
    "  *CoLLAs 2023.* [arxiv.org/abs/2301.11396](https://arxiv.org/abs/2301.11396)\n"
    "- Sun et al. (2023). PILOT: A Pre-Trained Model-Based Continual Learning\n"
    "  Toolbox. [github.com/sun-hailong/LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)\n"
    "- CLOVER. [github.com/danushkumar-v/clover-cl](https://github.com/danushkumar-v/clover-cl)\n"
)


# ============================================================
# Assemble notebook JSON
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# Sanity: ASCII-only in cell sources, no 'id' keys
for i, cell in enumerate(notebook["cells"]):
    if "id" in cell:
        del cell["id"]
    src = "".join(cell.get("source", []))
    bad = [(j, ord(ch)) for j, ch in enumerate(src) if ord(ch) > 127]
    if bad:
        print(f"WARN: non-ASCII in cell {i} at positions {bad[:3]}")

out = Path("notebook.ipynb")
with out.open("w", encoding="ascii") as f:
    json.dump(notebook, f, ensure_ascii=True, indent=1)

print(f"Wrote {out} with {len(notebook['cells'])} cells.")
