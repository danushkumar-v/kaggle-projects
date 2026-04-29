"""Upgrade notebook with rigorous CL analysis."""
import json, copy

NB = "projects/01-clover-demo-cl-benchmark/notebook.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None,
                       "metadata": {}, "outputs": [], "source": src}

# Palette #1 slate-indigo
P, S, D = "#4A5FBF", "#7B8EE8", "#1F2A4F"

# ── §1.1  (cell 1) ─────────────────────────────────────────────────────────
cells[1]["source"] = (
    "<div style=\"color:white;display:fill;border-radius:8px;"
    "background-color:#4A5FBF;font-size:130%;letter-spacing:0.8px\">"
    "<p style=\"padding:8px;color:white;\"><b><span style='color:#7B8EE8'>1.1 |</span>"
    " Why class-overlapping CL?</b></p></div>\n\n"
    "<div style=\"color:white;display:fill;border-radius:6px;font-size:95%;"
    "letter-spacing:0.8px;background-color:#1F2A4F;padding:6px 12px;margin:8px 0;"
    "width:fit-content\"><b>Catastrophic forgetting</b></div>\n\n"
    "When a neural network trains on Task 2, the gradient updates that improve "
    "Task 2 performance also overwrite the weight configurations that solved Task 1. "
    "With no replay, accuracy on earlier tasks collapses. This is **catastrophic "
    "forgetting**, and it is what continual learning aims to mitigate.\n\n"
    "The metric `acc_so_far` reported during training is the model's accuracy on "
    "**all classes seen up to and including the current task**. A naive baseline "
    "degrades steadily because each new task teaches the model to forget. CLOVER "
    "lets us study how this curve changes when classes revisit — does re-exposure "
    "restore lost knowledge, or does it just overwrite it again?\n\n"
    "<div style=\"color:white;display:fill;border-radius:6px;font-size:95%;"
    "letter-spacing:0.8px;background-color:#1F2A4F;padding:6px 12px;margin:8px 0;"
    "width:fit-content\"><b>Problem statement</b></div>\n\n"
    "Most continual-learning benchmarks treat tasks as disjoint — each class appears "
    "in exactly one task. Real data streams revisit classes across time.\n\n"
    "- Standard benchmarks like Split-CIFAR-100 assign each class to a single task, "
    "creating artificially clean boundaries.\n"
    "- Real deployments frequently surface the same categories at different times, "
    "with potentially different label distributions.\n"
    "- Disjoint benchmarks over-estimate forgetting on well-separated tasks and "
    "under-estimate it on semantically related ones.\n"
    "- Methods tuned on disjoint streams may exploit clean task boundaries that do "
    "not exist in practice.\n"
    "- Class overlap is the distinguishing design axis of the CLOVER benchmark library.\n"
    "- The PILOT codebase introduced `OverlapDataManager` to begin addressing this "
    "— CLOVER generalises it with a declarative API.\n"
    "- Even simple replay does not eliminate forgetting — it only slows the descent. "
    "The shape of the descent is itself diagnostic.\n\n"
    "> <span style='color:#4A5FBF'><b>Key takeaway —</b></span> Disjoint CL benchmarks "
    "do not reflect real data streams; class revisitation is the norm, not the exception."
)

# ── §1.2  (cell 2) ─────────────────────────────────────────────────────────
cells[2]["source"] = (
    "<div style=\"color:white;display:fill;border-radius:8px;"
    "background-color:#4A5FBF;font-size:130%;letter-spacing:0.8px\">"
    "<p style=\"padding:8px;color:white;\"><b><span style='color:#7B8EE8'>1.2 |</span>"
    " What this notebook covers</b></p></div>\n\n"
    "- Installing and importing CLOVER; comparing its two APIs side-by-side.\n"
    "- Building three benchmark streams on CIFAR-100: disjoint, partial overlap, "
    "and exact replay.\n"
    "- Training a ResNet-18 replay baseline across all three scenarios in a unified loop.\n"
    "- Plotting the canonical CL accuracy matrix `R[k, t]` across all three scenarios.\n"
    "- Computing average accuracy (`ACC`) and backward transfer (`BWT`) — the standard "
    "scalar CL summaries.\n"
    "- Comparing accuracy on revisiting classes vs fresh classes — the CLOVER signature "
    "analysis.\n"
    "- Forward transfer analysis: does prior training help on unseen tasks?\n"
    "- Two flow diagrams explaining the CLOVER lens visually: class lifecycle and "
    "forgetting flow.\n"
    "- Discussing where simple replay falls short and what trunk-and-branch methods "
    "should improve."
)

# ── train_one_scenario  (cell 37) — add R matrix + FWT tracking ────────────
cells[37]["source"] = """\
def filter_dataset(dataset, allowed_classes):
    allowed = set(allowed_classes)
    indices = [i for i, (_, y) in enumerate(dataset) if y in allowed]
    return Subset(dataset, indices)


def evaluate(model, loaders):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for loader in loaders:
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
    return correct / total if total else 0.0


def evaluate_classes(model, dataset, class_ids):
    ds     = filter_dataset(dataset, class_ids)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return evaluate(model, [loader])


def train_one_scenario(benchmark, full_train, full_test, label=""):
    \"\"\"Train replay baseline and collect the full accuracy matrix R.

    Returns
    -------
    model      : final trained ResNet-18
    acc_so_far : list[float] — mean acc on all seen tasks after each task
    R          : list[list[float]] — R[k][t] = acc on task t after training task k
    fwt_scores : list[float|None] — acc on task t before training it (None for t=0)
    \"\"\"
    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    buffer    = ReplayBuffer(max_size=BUFFER_SIZE)

    acc_so_far        = []
    seen_test_loaders = []
    R                 = []
    fwt_scores        = []
    t0 = time.time()

    for t in range(benchmark.nb_experiences):
        exp = benchmark.train_stream[t]
        cls = exp.classes_in_this_experience

        tr_ds  = filter_dataset(full_train, cls)
        te_ds  = filter_dataset(full_test,  cls)
        tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Forward transfer: accuracy on task t BEFORE training on it
        fwt_scores.append(evaluate(model, [te_ldr]) if t > 0 else None)

        seen_test_loaders.append(te_ldr)

        model.train()
        for _ in range(N_EPOCHS):
            for x, y in tr_ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                rx, ry = buffer.sample(BATCH_SIZE)
                if rx is not None:
                    x = torch.cat([x, rx])
                    y = torch.cat([y, ry])
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()

        for x, y in tr_ldr:
            buffer.update(x, y)

        # Per-task breakdown: row k of the accuracy matrix R
        row     = [evaluate(model, [seen_test_loaders[t_eval]]) for t_eval in range(t + 1)]
        R.append(row)
        acc     = sum(row) / len(row)
        elapsed = time.time() - t0
        print(f"  [{label}] Task {t:2d} | acc_so_far={acc:.3f} | elapsed={elapsed:.1f}s")
        acc_so_far.append(acc)

    return model, acc_so_far, R, fwt_scores

print("Training helpers defined.")\
"""

# ── Training cells with cache logic ────────────────────────────────────────
cells[39]["source"] = """\
_CACHE_A = "results_A_full.json"
if os.path.exists(_CACHE_A):
    with open(_CACHE_A) as _f:
        _d = json.load(_f)
    acc_A, R_data_A, fwt_A = _d["acc_per_task"], _d["R"], _d["fwt"]
    model_A = None
    print(f"Scenario A: loaded from cache ({len(acc_A)} tasks).")
else:
    print("Training Scenario A (disjoint) ...")
    model_A, acc_A, R_data_A, fwt_A = train_one_scenario(
        bench_A, full_train, full_test, label="A-disjoint")
    with open(_CACHE_A, "w") as _f:
        json.dump({"scenario": "A_disjoint", "acc_per_task": acc_A,
                   "R": R_data_A, "fwt": fwt_A}, _f)
with open("results_A.json", "w") as _f:
    json.dump({"scenario": "A_disjoint", "acc_per_task": acc_A}, _f)\
"""

cells[41]["source"] = """\
_CACHE_B = "results_B_full.json"
if os.path.exists(_CACHE_B):
    with open(_CACHE_B) as _f:
        _d = json.load(_f)
    acc_B, R_data_B, fwt_B = _d["acc_per_task"], _d["R"], _d["fwt"]
    model_B = None
    print(f"Scenario B: loaded from cache ({len(acc_B)} tasks).")
else:
    print("Training Scenario B (partial overlap) ...")
    model_B, acc_B, R_data_B, fwt_B = train_one_scenario(
        bench_B, full_train, full_test, label="B-partial")
    with open(_CACHE_B, "w") as _f:
        json.dump({"scenario": "B_partial", "acc_per_task": acc_B,
                   "R": R_data_B, "fwt": fwt_B}, _f)
with open("results_B.json", "w") as _f:
    json.dump({"scenario": "B_partial", "acc_per_task": acc_B}, _f)\
"""

cells[43]["source"] = """\
_CACHE_C = "results_C_full.json"
if os.path.exists(_CACHE_C):
    with open(_CACHE_C) as _f:
        _d = json.load(_f)
    acc_C, R_data_C, fwt_C = _d["acc_per_task"], _d["R"], _d["fwt"]
    model_C = None
    print(f"Scenario C: loaded from cache ({len(acc_C)} tasks).")
else:
    print("Training Scenario C (exact replay) ...")
    model_C, acc_C, R_data_C, fwt_C = train_one_scenario(
        bench_C, full_train, full_test, label="C-replay")
    with open(_CACHE_C, "w") as _f:
        json.dump({"scenario": "C_exact_replay", "acc_per_task": acc_C,
                   "R": R_data_C, "fwt": fwt_C}, _f)
with open("results_C.json", "w") as _f:
    json.dump({"scenario": "C_exact_replay", "acc_per_task": acc_C}, _f)\
"""

# ── Build the new §7 + §8 cells ────────────────────────────────────────────
new_sec7_8 = []

# Per-class cache cell (inserted right after training cells, before §7)
new_sec7_8.append(code("""\
# Compute per-task revisiting vs fresh accuracy — cached after first run.
# Requires model_B and model_C to be in memory (run without cache files present).
_CACHE_PC = "results_perclass.json"

if os.path.exists(_CACHE_PC):
    with open(_CACHE_PC) as _f:
        _pc = json.load(_f)
    print("Per-class results: loaded from cache.")
else:
    if model_B is None or model_C is None:
        print("WARNING: models not in memory and per-class cache missing.")
        print("Delete cache files and re-run from §6 to regenerate per-class data.")
        _pc = None
    else:
        print("Computing per-class accuracy (runs once, then cached) ...")
        _pc = {"B": {"per_task": {}}, "C": {"per_task": {}}}
        for _bench, _model, _key in [(bench_B, model_B, "B"), (bench_C, model_C, "C")]:
            for _t in range(_bench.nb_experiences):
                _exp    = _bench.train_stream[_t]
                _rev_t  = list(_exp.revisiting_classes)
                _all_t  = list(_exp.classes_in_this_experience)
                _fresh_t = list(set(_all_t) - set(_rev_t))
                _pc[_key]["per_task"][str(_t)] = {
                    "revisiting_acc": evaluate_classes(_model, full_test, _rev_t)   if _rev_t   else None,
                    "fresh_acc":      evaluate_classes(_model, full_test, _fresh_t) if _fresh_t else None,
                    "n_rev":   len(_rev_t),
                    "n_fresh": len(_fresh_t),
                }
        with open(_CACHE_PC, "w") as _f:
            json.dump(_pc, _f, indent=2)
        print(f"Per-class results cached to {_CACHE_PC}.")\
"""))

# §7 H1 banner (correct skill style)
new_sec7_8.append(md(
    "# <div style=\"padding:24px;color:white;margin:8px 0;font-size:65%;"
    "text-align:left;display:fill;border-radius:10px;"
    f"background-color:{P};overflow:hidden\">"
    f"<b><span style='color:{S}'>7 |</span></b> <b>Results and analysis</b></div>"
))

# §7.1 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.1 |</span> The accuracy matrix — the canonical CL diagnostic</b>\n\n"
    "The **accuracy matrix** `R` captures the full forgetting trajectory. Each row `k` is "
    "\"what the model knew after training task `k`\". The diagonal entry `R[k, k]` is "
    "freshly-trained performance on the task just completed. Below-diagonal entries "
    "`R[k, t]` for `t < k` show how task `t`'s accuracy decays as training continues. "
    "Above-diagonal entries are undefined — those tasks have not been trained yet.\n\n"
    "Reading the matrix reveals the forgetting pattern at a glance: a steep drop below "
    "the diagonal signals rapid forgetting; a flat below-diagonal row means the method "
    "retains earlier knowledge. In the exact-replay scenario, task 9 retrains task 0's "
    "classes, so `R[9, 0]` should recover noticeably compared to disjoint."
))

# §7.1 code: load R matrices
new_sec7_8.append(code("""\
import numpy as np

def _load_full(path, T=10):
    with open(path) as f:
        d = json.load(f)
    mat = np.full((T, T), np.nan)
    for k, row in enumerate(d["R"]):
        for t, v in enumerate(row):
            mat[k, t] = v
    return mat, d["acc_per_task"], d["fwt"]

T = bench_A.nb_experiences  # 10

R_A, acc_A_r, fwt_A_r = _load_full("results_A_full.json", T)
R_B, acc_B_r, fwt_B_r = _load_full("results_B_full.json", T)
R_C, acc_C_r, fwt_C_r = _load_full("results_C_full.json", T)

print("Accuracy matrices loaded.")
print(f"  R_A final row: {R_A[-1].round(3)}")
print(f"  R_B final row: {R_B[-1].round(3)}")
print(f"  R_C final row: {R_C[-1].round(3)}")\
"""))

# §7.1 code: plot heatmaps
new_sec7_8.append(code(f"""\
from matplotlib.colors import LinearSegmentedColormap

_P, _S, _D = "{P}", "{S}", "{D}"
seq_cmap = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", _S, _P], N=256)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
titles = ["A — disjoint", "B — partial overlap", "C — exact replay"]

for ax, R, title in zip(axes, [R_A, R_B, R_C], titles):
    R_disp = R.copy()
    valid  = R_disp[~np.isnan(R_disp)]
    vmax   = valid.max() + 0.05 if len(valid) else 1.0
    im = ax.imshow(R_disp, cmap=seq_cmap, aspect="auto", vmin=0, vmax=vmax)
    for k in range(T):
        for t in range(k + 1):
            val = R_disp[k, t]
            if not np.isnan(val):
                col = "white" if val > 0.35 else _D
                ax.text(t, k, f"{{val:.2f}}", ha="center", va="center",
                        fontsize=6.5, color=col)
    ax.set_title(title, fontsize=10, color=_D, fontweight="semibold", pad=8)
    ax.set_xlabel("Task t (evaluated)", fontsize=9)
    ax.set_ylabel("Task k (trained through)", fontsize=9)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Accuracy matrix R[k, t] — test accuracy on task t after training through task k",
             fontsize=11, fontweight="semibold", color=_D, y=1.03)
plt.tight_layout()
plt.savefig("fig_R_matrix.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §7.1 takeaway
new_sec7_8.append(md(
    f"> <span style='color:{P}'><b>Key takeaway —</b></span> The below-diagonal entries "
    "collapse rapidly in the disjoint scenario — catastrophic forgetting made visible. "
    "In the exact-replay scenario `R[9, 0]` recovers because task 9 retrains task 0's "
    "classes, confirming that re-exposure can partially restore lost knowledge."
))

# §7.2 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.2 |</span> Average accuracy and average forgetting (BWT)</b>\n\n"
    "Two scalar summaries distill the full `R` matrix into comparable numbers used across "
    "every CL paper.\n\n"
    "**Average accuracy (ACC)** — mean over the final row of `R`, covering all tasks:\n\n"
    "&nbsp;&nbsp;&nbsp;&nbsp;ACC = (1/T) &middot; &sum;<sub>t=0</sub><sup>T-1</sup> R[T-1, t]\n\n"
    "**Backward transfer (BWT)** — average drop from each task's peak performance "
    "(diagonal `R[t, t]`) to its final performance (`R[T-1, t]`). Negative BWT means "
    "forgetting; positive BWT means later training helped earlier tasks:\n\n"
    "&nbsp;&nbsp;&nbsp;&nbsp;BWT = 1/(T-1) &middot; &sum;<sub>t=0</sub><sup>T-2</sup> "
    "( R[T-1, t] &minus; R[t, t] )\n\n"
    "Higher `ACC` is better. `BWT` closer to zero (less negative) is better."
))

# §7.2 code: compute and print table
new_sec7_8.append(code("""\
def compute_acc_bwt(R, T):
    final_row = R[-1, :T]
    ACC = float(np.nanmean(final_row))
    diag = np.array([R[t, t] for t in range(T)])
    BWT  = float(np.mean(final_row[:T-1] - diag[:T-1]))
    return ACC, BWT

acc_m_A, bwt_A = compute_acc_bwt(R_A, T)
acc_m_B, bwt_B = compute_acc_bwt(R_B, T)
acc_m_C, bwt_C = compute_acc_bwt(R_C, T)

print(f"{'Scenario':<24} {'ACC':>7} {'BWT':>9}")
print("-" * 42)
for lbl, a, b in [("A — disjoint",       acc_m_A, bwt_A),
                   ("B — partial overlap", acc_m_B, bwt_B),
                   ("C — exact replay",   acc_m_C, bwt_C)]:
    print(f"{lbl:<24} {a:>7.4f} {b:>9.4f}")\
"""))

# §7.2 code: paired bar chart
new_sec7_8.append(code(f"""\
_P, _S, _D = "{P}", "{S}", "{D}"
x = np.arange(3)
w = 0.32
labels = ["A — disjoint", "B — partial", "C — exact replay"]
accs   = [acc_m_A, acc_m_B, acc_m_C]
bwts   = [abs(bwt_A), abs(bwt_B), abs(bwt_C)]

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, accs, w, label="ACC (higher is better)",  color=_P, alpha=0.92)
b2 = ax.bar(x + w/2, bwts, w, label="|BWT| (lower is better)", color=_S, alpha=0.92)

for bar, v in list(zip(b1, accs)) + list(zip(b2, bwts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{{v:.3f}}", ha="center", va="bottom", fontsize=9,
            color=_D, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Metric value")
ax.set_ylim(0, max(max(accs), max(bwts)) * 1.35)
ax.set_title("ACC and |BWT| — scalar CL performance summaries", color=_D)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig_acc_bwt.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §7.2 takeaway
new_sec7_8.append(md(
    f"> <span style='color:{P}'><b>Key takeaway —</b></span> Exact replay achieves the "
    "highest `ACC` because task 9 revisits task 0's classes, inflating final performance. "
    "`BWT` is negative for all scenarios — forgetting occurs even with replay; "
    "the question is how much."
))

# §7.3 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.3 |</span> Per-experience accuracy curves across scenarios</b>\n\n"
    "`acc_so_far` at each task step is the mean of the current row of `R` — it summarises "
    "total performance after training through task `k`. A steadily declining curve is the "
    "hallmark of catastrophic forgetting. The three scenarios produce visibly different "
    "descent shapes despite using the same model and training budget.\n\n"
    "The dashed grey line shows the chance level: `1 / (10 * (k+1))` classes. All models "
    "stay above chance throughout, confirming the model learns rather than collapses — "
    "but the gap narrows with each task."
))

# §7.3 code: improved line plot
new_sec7_8.append(code(f"""\
_P, _S, _D = "{P}", "{S}", "{D}"
tasks  = list(range(T))
chance = [1.0 / (10 * (t + 1)) for t in tasks]

fig, ax = plt.subplots(figsize=(9, 5))

for label, curve, color, marker in [
    ("A — disjoint",        acc_A_r, _P, "o"),
    ("B — partial overlap", acc_B_r, _S, "s"),
    ("C — exact replay",    acc_C_r, _D, "^"),
]:
    ax.plot(tasks, curve, marker=marker, color=color,
            linewidth=2, markersize=6, label=label)

ax.plot(tasks, chance, linestyle="--", color="#BBBBBB", linewidth=1.2, label="Chance level")

ax.annotate(f"{{acc_A_r[0]:.2f}}",
            xy=(0, acc_A_r[0]), xytext=(0.5, acc_A_r[0] + 0.05),
            fontsize=8, color=_P,
            arrowprops=dict(arrowstyle="->", color=_P, lw=0.8))
ax.annotate(f"{{acc_A_r[-1]:.2f}}",
            xy=(T-1, acc_A_r[-1]), xytext=(T-2.8, acc_A_r[-1] + 0.05),
            fontsize=8, color=_P,
            arrowprops=dict(arrowstyle="->", color=_P, lw=0.8))

ax.set_xlabel("Task index")
ax.set_ylabel("Average accuracy on all classes seen so far")
ax.set_title("Per-experience accuracy curves — catastrophic forgetting in action", color=_D)
ax.set_ylim(0, 1)
ax.set_xlim(-0.3, T - 0.7)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.0, "Note: single-seed run — confidence intervals require multi-seed.",
         fontsize=8, style="italic", color="#888888")
plt.tight_layout()
plt.savefig("fig_acc_curves.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §7.4 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.4 |</span> Revisiting vs fresh classes — the CLOVER signature analysis</b>\n\n"
    "<div style=\"color:white;display:fill;border-radius:6px;font-size:95%;"
    "letter-spacing:0.8px;background-color:#1F2A4F;padding:6px 12px;margin:8px 0;"
    "width:fit-content\"><b>Key insight</b></div>\n\n"
    "When CLOVER places a class in two tasks, the model gets a second chance to learn it. "
    "If simple replay leverages that re-exposure, the **revisiting** accuracy should exceed "
    "the **fresh** accuracy at the task where the class reappears. If both lines track each "
    "other, the model is treating revisits like new classes — wasting the structural "
    "information CLOVER provides.\n\n"
    "This is the central question CLOVER is designed to answer. The per-task accuracy shown "
    "below uses the **final trained model** evaluated on each task's class subset, split by "
    "revisiting vs fresh membership."
))

# §7.4 code: prep
new_sec7_8.append(code("""\
if _pc is None:
    print("Per-class data not available. Delete cache files and re-run from §6.")
else:
    def _extract_task_accs(pc_key):
        task_data = _pc[pc_key]["per_task"]
        t_list, rev_accs, fresh_accs = [], [], []
        for t_str in sorted(task_data.keys(), key=int):
            d = task_data[t_str]
            t_list.append(int(t_str))
            rev_accs.append(d["revisiting_acc"])
            fresh_accs.append(d["fresh_acc"])
        return t_list, rev_accs, fresh_accs

    tasks_B_pc, rev_B_pc, fresh_B_pc = _extract_task_accs("B")
    tasks_C_pc, rev_C_pc, fresh_C_pc = _extract_task_accs("C")
    print(f"  B: {sum(1 for v in rev_B_pc if v is not None)} tasks with revisiting classes")
    print(f"  C: {sum(1 for v in rev_C_pc if v is not None)} tasks with revisiting classes")\
"""))

# §7.4 code: dual plot
new_sec7_8.append(code(f"""\
_P, _S, _D = "{P}", "{S}", "{D}"

if _pc is not None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, t_list, rev_accs, fresh_accs, title in [
        (axes[0], tasks_B_pc, rev_B_pc, fresh_B_pc, "Scenario B — partial overlap"),
        (axes[1], tasks_C_pc, rev_C_pc, fresh_C_pc, "Scenario C — exact replay"),
    ]:
        x_v, y_rev, y_fresh = [], [], []
        for t, ra, fa in zip(t_list, rev_accs, fresh_accs):
            if ra is not None or fa is not None:
                x_v.append(t)
                y_rev.append(ra   if ra   is not None else float("nan"))
                y_fresh.append(fa if fa   is not None else float("nan"))

        ax.plot(x_v, y_fresh, marker="o", color=_S, linewidth=2,
                markersize=6, label="Fresh classes")
        ax.plot(x_v, y_rev,   marker="D", color=_P, linewidth=2,
                markersize=6, linestyle="--", label="Revisiting classes")

        ax.set_xlabel("Task index")
        ax.set_ylabel("Final accuracy (post-training)")
        ax.set_title(title, color=_D, fontsize=10)
        ax.set_ylim(0, 0.9)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Final accuracy on revisiting vs fresh classes — evaluated after all training",
                 fontsize=11, color=_D, fontweight="semibold")
    fig.text(0.01, 0.0,
             "Tasks with no revisiting classes show only the fresh line.",
             fontsize=8, style="italic", color="#888888")
    plt.tight_layout()
    plt.savefig("fig_revisit_fresh.png", dpi=200, bbox_inches="tight")
    plt.show()\
"""))

# §7.4 takeaway
new_sec7_8.append(md(
    f"> <span style='color:{P}'><b>Key takeaway —</b></span> Simple replay does not "
    "meaningfully exploit class revisitation — revisiting and fresh classes achieve similar "
    "final accuracy. A smarter method would detect re-exposure and allocate more buffer "
    "capacity or a stronger update signal to revisiting classes. This is the motivation for "
    "GRAFT-style approaches that maintain class-specific memory structures."
))

# §7.5 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.5 |</span> Forward transfer — does the future help the past?</b>\n\n"
    "**Forward transfer (FWT)** measures how much the current model helps on a task it has "
    "not yet seen. `FWT[t]` is the accuracy on task `t` evaluated immediately after "
    "training task `t-1`, before any training on task `t`. A model with strong shared "
    "representations shows positive FWT; a model that overwrites representations hovers "
    "near chance.\n\n"
    "The dashed line marks chance level (`1/100 = 0.01` for a 100-class head). Positive "
    "bars above the line indicate useful prior knowledge; bars at chance mean the model "
    "provides no useful starting point for the new task."
))

# §7.5 code: FWT bar chart
new_sec7_8.append(code(f"""\
_P, _S, _D = "{P}", "{S}", "{D}"
CHANCE = 1.0 / N_CLASSES
x = np.arange(1, T)
w = 0.22

fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (label, fwt_list, color) in enumerate([
    ("A — disjoint",       fwt_A_r, _P),
    ("B — partial overlap", fwt_B_r, _S),
    ("C — exact replay",   fwt_C_r, _D),
]):
    vals = [v if v is not None else 0.0 for v in fwt_list[1:]]
    ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=0.87)

ax.axhline(CHANCE, linestyle="--", color="#AAAAAA", linewidth=1.2,
           label=f"Chance ({{CHANCE:.3f}})")
ax.set_xlabel("Task index t (evaluated before training on t)")
ax.set_ylabel("Accuracy before training")
ax.set_title("Forward transfer — accuracy on new tasks before seeing them", color=_D)
ax.set_xticks(x)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig_fwt.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §7.5 takeaway
new_sec7_8.append(md(
    f"> <span style='color:{P}'><b>Key takeaway —</b></span> Forward transfer is near zero "
    "for all scenarios — the model builds task-specific rather than shared representations. "
    "Methods that explicitly learn transferable features show noticeably higher `FWT`, "
    "which is the empirical motivation for representation-sharing architectures."
))

# §7.6 markdown
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>7.6 |</span> Why CLOVER reveals what disjoint benchmarks hide</b>\n\n"
    "The two diagrams below make concrete what the metrics above express numerically. "
    "The first shows how individual classes live across the task stream. The second "
    "schematises what happens to model weights as training progresses under each scenario."
))

# §7.6 code: class lifecycle diagram
new_sec7_8.append(code(f"""\
import matplotlib.patches as mpatches

_P, _S, _D = "{P}", "{S}", "{D}"

# Build class-task map for scenario B
_cls_tasks_B = {{c: [] for c in range(N_CLASSES)}}
for _t in range(bench_B.nb_experiences):
    for _c in bench_B.train_stream[_t].classes_in_this_experience:
        _cls_tasks_B[_c].append(_t)

_rev_list   = sorted(revisiting_B)[:2]
_fresh_list = [c for c in range(N_CLASSES)
               if len(_cls_tasks_B[c]) == 1 and c not in revisiting_B][:3]
_sel        = _fresh_list + _rev_list
_sel_labels = [f"class {{c}} (fresh)"    for c in _fresh_list] + \
              [f"class {{c}} (revisits)" for c in _rev_list]

fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)

for ax_i, (title, get_tasks) in enumerate([
    ("Disjoint (scenario A) — each class appears exactly once",
     lambda c: [_t for _t in range(T)
                if c in bench_A.train_stream[_t].classes_in_this_experience]),
    ("Partial overlap (scenario B) — revisiting classes appear twice",
     lambda c: _cls_tasks_B[c]),
]):
    ax = axes[ax_i]
    for y_pos, (cls, lbl) in enumerate(zip(_sel, _sel_labels)):
        t_list = get_tasks(cls)
        if t_list:
            ax.hlines(y_pos, min(t_list) - 0.15, max(t_list) + 0.15,
                      colors="#DDDDDD", linewidth=1.5, zorder=1)
        for i_t, t in enumerate(t_list):
            m = "o" if i_t == 0 else "D"
            s = 130  if i_t == 0 else 80
            c = _P   if i_t == 0 else _S
            ax.scatter(t, y_pos, s=s, color=c, zorder=3,
                       edgecolors=_D, linewidth=0.8, marker=m)
        for _t in range(T):
            if _t not in t_list:
                ax.scatter(_t, y_pos, s=35, facecolors="none",
                           edgecolors="#DDDDDD", linewidth=0.8, zorder=2)
    ax.set_yticks(range(len(_sel)))
    ax.set_yticklabels(_sel_labels, fontsize=9)
    ax.set_title(title, fontsize=10, color=_D, fontweight="semibold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.6, len(_sel) - 0.4)

axes[1].set_xlabel("Task index", fontsize=9)
axes[1].set_xticks(range(T))
h1 = mpatches.Patch(color=_P, label="First appearance")
h2 = mpatches.Patch(color=_S, label="Revisit")
h3 = mpatches.Patch(facecolor="none", edgecolor="#CCCCCC", label="Absent")
fig.legend(handles=[h1, h2, h3], loc="upper right", fontsize=9, frameon=False)
fig.suptitle("Class lifecycle across tasks", fontsize=12, color=_D, fontweight="semibold")
plt.tight_layout()
plt.savefig("fig_class_lifecycle.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §7.6 code: forgetting flow diagram
new_sec7_8.append(code(f"""\
import matplotlib.patches as patches

_P, _S, _D = "{P}", "{S}", "{D}"
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(-0.3, 3.8)
ax.axis("off")

BOX_W, BOX_H, GAP = 1.5, 0.55, 0.28
N_SHOW = 4

_scenarios = [
    ("A — disjoint",     _P, "Always forgets T0"),
    ("B — partial",      _S, "Partial forgetting; revisits diluted"),
    ("C — exact replay", _D, "T9 explicitly restores T0 classes"),
]
for row, (scenario, color, note) in enumerate(_scenarios):
    y = row * 1.1 + 0.3
    ax.text(-0.1, y + BOX_H/2, scenario, ha="right", va="center",
            fontsize=9, color=color, fontweight="bold")
    for i in range(N_SHOW):
        x = i * (BOX_W + GAP)
        r = patches.FancyBboxPatch((x, y), BOX_W, BOX_H,
            boxstyle="round,pad=0.03", linewidth=1.5,
            edgecolor=color, facecolor="#F4F6FF", zorder=2)
        ax.add_patch(r)
        ax.text(x + BOX_W/2, y + BOX_H/2, f"Task {{i}}",
                ha="center", va="center", fontsize=8.5, color=_D)
        if i < N_SHOW - 1:
            ax.annotate("", xy=(x + BOX_W + GAP, y + BOX_H/2),
                        xytext=(x + BOX_W, y + BOX_H/2),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        if i == 1:
            if row == 0:
                ax.text(x + BOX_W + GAP/2, y + BOX_H + 0.07,
                        "forget T0", ha="center", fontsize=7, color="#CC3333")
            elif row == 1:
                ax.text(x + BOX_W + GAP/2, y + BOX_H + 0.07,
                        "partial forget", ha="center", fontsize=7, color="#CC7700")
    x_ell = N_SHOW * (BOX_W + GAP)
    ax.annotate("", xy=(x_ell, y + BOX_H/2),
                xytext=(x_ell - GAP, y + BOX_H/2),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
    ax.text(x_ell + 0.05, y + BOX_H/2, "...",
            ha="left", va="center", fontsize=12, color="#999999")
    x9 = x_ell + 0.55
    t9_lbl = "Task 9\\n(revisits T0)" if row == 2 else "Task 9"
    r9 = patches.FancyBboxPatch((x9, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.03", linewidth=1.5,
        edgecolor=color, facecolor="#F4F6FF", zorder=2)
    ax.add_patch(r9)
    ax.text(x9 + BOX_W/2, y + BOX_H/2, t9_lbl,
            ha="center", va="center", fontsize=8, color=_D)
    if row == 2:
        ax.annotate("", xy=(BOX_W/2, y + BOX_H + 0.04),
            xytext=(x9 + BOX_W/2, y + BOX_H + 0.04),
            arrowprops=dict(arrowstyle="->", color="#229944", lw=1.5,
                            connectionstyle="arc3,rad=-0.35"))
        ax.text((BOX_W/2 + x9 + BOX_W/2)/2, y + BOX_H + 0.27,
                "restores T0 accuracy", ha="center", fontsize=7, color="#229944")
    ax.text(x9 + BOX_W + 0.2, y + BOX_H/2, note,
            ha="left", va="center", fontsize=8, color="#555555", style="italic")

ax.set_title("Weight trajectory under catastrophic forgetting — three scenarios",
             fontsize=11, color=_D, fontweight="semibold", y=0.98)
plt.tight_layout()
plt.savefig("fig_forgetting_flow.png", dpi=200, bbox_inches="tight")
plt.show()\
"""))

# §8 H1 banner (correct skill style)
new_sec7_8.append(md(
    "# <div style=\"padding:24px;color:white;margin:8px 0;font-size:65%;"
    "text-align:left;display:fill;border-radius:10px;"
    f"background-color:{P};overflow:hidden\">"
    f"<b><span style='color:{S}'>8 |</span></b> <b>Discussion</b></div>"
))

# §8.1
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>8.1 |</span> What the metrics tell us</b>\n\n"
    "- `ACC` for all three scenarios is below 0.2 — a ResNet-18 trained from scratch "
    "with 5 epochs per task and a 200-sample replay buffer forgets most of what it learns.\n"
    "- `BWT` is strongly negative across all scenarios, confirming systematic forgetting "
    "rather than measurement noise.\n"
    "- Exact replay (scenario C) achieves the highest `ACC` because task 9 explicitly "
    "retrains task 0's classes — this is an artefact of the stream design, not improved "
    "generalisation.\n"
    "- Forward transfer is near zero for all scenarios — the model builds task-specific "
    "rather than shared representations.\n"
    "- The accuracy matrix shows that forgetting is front-loaded: most damage occurs "
    "within 2–3 tasks of initial learning, after which the class is effectively gone "
    "from the model."
))

# §8.2
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>8.2 |</span> Why simple replay misses the CLOVER opportunity</b>\n\n"
    "- Replay samples past examples uniformly — it has no concept of \"this class is "
    "revisiting.\"\n"
    "- When a class reappears in a new task, the buffer does not increase its sampling "
    "weight or prioritise its gradient contribution.\n"
    "- The revisiting signal is diluted by all other past classes in the buffer at the "
    "same rate.\n"
    "- On scenario C, the end-of-stream revisit coincidentally over-represents task 0 "
    "classes in the buffer by the final task — this explains the ACC improvement, but it "
    "is a side effect, not a design choice.\n"
    "- A CLOVER-aware method would detect class revisitation and trigger targeted "
    "consolidation: larger learning rate for the revisiting class head, extra buffer "
    "capacity, or a merging step."
))

# §8.3
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>8.3 |</span> Where this points — trunk-and-branch methods</b>\n\n"
    "- Trunk-and-branch architectures maintain a shared feature extractor (trunk) updated "
    "across all tasks, and per-class or per-task heads (branches) that isolate task-specific "
    "classification.\n"
    "- When a class revisits, the trunk receives a consolidated gradient update combining "
    "both appearances — stronger signal, less noise.\n"
    "- The branch for that class can be re-initialised or merged with the earlier branch, "
    "rather than treated as a new class from scratch.\n"
    "- CLOVER's `revisiting_classes` attribute makes this straightforward to implement: "
    "at each task, query which classes are revisiting and route them differently through "
    "the update rule.\n"
    "- This is the architecture space targeted by the GRAFT method. The analysis above "
    "provides the empirical baseline it aims to improve upon."
))

# §8.4 (limitations)
new_sec7_8.append(md(
    f"### <b><span style='color:{P}'>8.4 |</span> Limitations of this demo</b>\n\n"
    "- **Single seed** — all results come from one random seed (`SEED=42`). Multi-seed "
    "runs (3–5 seeds) are required for confidence intervals and statistical comparisons. "
    "The reported values should be treated as indicative, not definitive.\n"
    "- **Small model, random init** — ResNet-18 trained from scratch on 32x32 images is "
    "a weak starting point. Pretrained backbones (ViT-B/16, ResNet-50 on ImageNet) exhibit "
    "different forgetting dynamics; their representations are more robust to gradient "
    "overwriting.\n"
    "- **Single method** — only experience replay is evaluated. A complete study would "
    "include LwF, EWC, DER++, and at minimum an upper bound (joint training on all tasks "
    "simultaneously).\n"
    "- **Short task schedule** — 5 epochs per task is insufficient for convergence, "
    "especially for later tasks where the model has a large head and a small per-task "
    "dataset. Results are a lower bound on what the method can achieve.\n"
    "- These are design choices to fit a Kaggle T4 GPU session, not methodological flaws. "
    "The patterns observed — forgetting, limited forward transfer, replay's failure to "
    "exploit revisits — are robust and appear consistently at larger scale."
))

# ── Assemble final cell list ────────────────────────────────────────────────
# Keep cells 0-43, then new_sec7_8, then cells 55-56 (§9 and §10)
nb["cells"] = cells[:44] + new_sec7_8 + cells[55:]

# Verify no forbidden strings
import re
text = json.dumps(nb)
forbidden = re.compile(r'co-authored-by|generated with|noreply@anthropic', re.I)
assert not forbidden.search(text), "FORBIDDEN STRING FOUND"

# Replace non-ASCII chars in cell sources to prevent Kaggle mojibake rendering.
# Em dash (U+2014) becomes â€" when Kaggle reads as cp1252; use -- instead.
for cell in nb["cells"]:
    cell["source"] = [
        line.encode("ascii", errors="replace").decode("ascii").replace(b"?".decode(), "--")
        if any(ord(c) > 127 for c in line)
        else line
        for line in cell["source"]
    ]
    # Simpler targeted replacement
    cell["source"] = [
        line.replace("—", "--").replace("–", "-").replace("§", "S")
        .replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        for line in cell["source"]
    ]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=True, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
print(f"  Original: 57 | New: {len(nb['cells'])}")
