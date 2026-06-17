"""
Stage 1b — Multi-Model Behavioral Pre-screening
================================================
目的：验证 discourse 模板优势是否在 Pythia 家族跨尺度保持，
      为论文提供 scaling 数据点。

运行方式：
  python run_multimodel_stage1.py
  （顺序加载 Pythia-410M 和 Pythia-2.8B，卸载后再加载下一个）

VRAM 估算（bfloat16）：
  pythia-410m : ~0.82 GB
  pythia-2.8b : ~5.6  GB
  两模型顺序运行，峰值 VRAM 由最大模型决定（~5.6 GB）

已有 Pythia-1.4B 结果来源：../stage1_prescreen/prescreen_results.json
新结果保存：
  results_410m.json
  results_2b8.json
  fig_scaling.png
"""

import json
import gc
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
PRESCREEN_1B4 = ROOT / "stage1_prescreen" / "prescreen_results.json"
SAVE_DIR      = Path(__file__).parent

# ── models to run ─────────────────────────────────────────────────────────────
# Set SKIP_2B8 = True if GPU VRAM < 6 GB
SKIP_2B8 = False

MODELS = [
    ("410m",  "pythia-410m"),
    ("2b8",   "pythia-2.8b"),
]

# ── templates (same as Stage 1) ───────────────────────────────────────────────
TEMPLATES = {
    "discourse": lambda p: f"Context: {p}\nConclusion: ",
    "bare":      lambda p: p + " ",
}

PRAG_MARGIN_THRESHOLD = 0.0   # prag_margin > 0 ⟹ success


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> HookedTransformer:
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()
    return model


def unload_model(model: HookedTransformer):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def score_conditional(model, prefix: str, hypothesis: str) -> float:
    """Per-token average log-prob of hypothesis given prefix. Identical to Stage 1."""
    prefix_ids = model.to_tokens(prefix, prepend_bos=True)
    hyp_ids    = model.to_tokens(hypothesis, prepend_bos=False)
    full_ids   = torch.cat([prefix_ids, hyp_ids], dim=1)
    with torch.no_grad():
        logits = model(full_ids)
    p_len      = prefix_ids.shape[1]
    h_len      = hyp_ids.shape[1]
    logits_hyp = logits[0, p_len - 1 : p_len + h_len - 1, :]
    log_probs  = F.log_softmax(logits_hyp, dim=-1)
    token_lp   = log_probs[torch.arange(h_len), hyp_ids[0]]
    return (token_lp.sum() / h_len).item()


def run_model_prescreen(model, items: list, model_tag: str) -> dict:
    """
    Score all items under discourse and bare templates.
    Returns dict with per-item prag_margins and aggregate stats.
    """
    results_per_item = []

    for idx, item in enumerate(items):
        premise = item["premise"]
        hyps    = item["hypotheses"]  # keys: prag, impl, lit

        item_result = {"premise": premise, "templates": {}}
        for tname, prefix_fn in TEMPLATES.items():
            prefix = prefix_fn(premise)
            scores = {k: score_conditional(model, prefix, h) for k, h in hyps.items()}
            prag_margin = scores["prag"] - max(scores["impl"], scores["lit"])
            item_result["templates"][tname] = {
                "scores":      scores,
                "prag_margin": round(prag_margin, 4),
                "success":     prag_margin > PRAG_MARGIN_THRESHOLD,
            }
        results_per_item.append(item_result)

        if (idx + 1) % 25 == 0:
            print(f"  [{model_tag}] {idx+1}/{len(items)} items scored")
            gc.collect()
            torch.cuda.empty_cache()

    # ── aggregate stats ───────────────────────────────────────────────────────
    stats = {}
    for tname in TEMPLATES:
        margins  = [it["templates"][tname]["prag_margin"] for it in results_per_item]
        successes = [it["templates"][tname]["success"]    for it in results_per_item]
        stats[tname] = {
            "success_rate":     round(sum(successes) / len(successes), 4),
            "mean_prag_margin": round(float(np.mean(margins)), 4),
            "std_prag_margin":  round(float(np.std(margins)),  4),
            "n_success":        sum(successes),
            "n_total":          len(successes),
        }

    return {"model": model_tag, "stats": stats, "items": results_per_item}


# ── load 1.4B stats from existing prescreen_results.json ─────────────────────

def extract_1b4_stats(prescreen_path: Path) -> dict:
    """
    Extract discourse/bare prag_margin stats from the existing 1.4B results.
    prescreen_results.json uses 'conditions' key (not 'templates'), same score format.
    """
    with open(prescreen_path) as f:
        data = json.load(f)

    stats = {}
    for tname in ("discourse", "bare"):
        margins   = []
        successes = []
        for item in data:
            cond = item["conditions"].get(tname)
            if cond is None:
                continue
            pm = cond["prag"] - max(cond["impl"], cond["lit"])
            margins.append(pm)
            successes.append(pm > PRAG_MARGIN_THRESHOLD)
        stats[tname] = {
            "success_rate":     round(sum(successes) / len(successes), 4),
            "mean_prag_margin": round(float(np.mean(margins)), 4),
            "std_prag_margin":  round(float(np.std(margins)),  4),
            "n_success":        sum(successes),
            "n_total":          len(successes),
        }
    return {"model": "1b4", "stats": stats}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(all_stats: list, save_path: Path):
    """
    all_stats: list of dicts with keys 'model' and 'stats'
    stats structure: {template: {success_rate, mean_prag_margin, std_prag_margin, ...}}
    """
    model_labels = {"410m": "Pythia-410M", "1b4": "Pythia-1.4B", "2b8": "Pythia-2.8B"}
    colors       = {"discourse": "#DD8452", "bare": "#4C72B0"}

    models   = [s["model"] for s in all_stats]
    x        = np.arange(len(models))
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── subplot 1: success rate ───────────────────────────────────────────────
    ax = axes[0]
    for i, tname in enumerate(("discourse", "bare")):
        rates = [s["stats"][tname]["success_rate"] for s in all_stats]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=tname.capitalize(),
                      color=colors[tname], alpha=0.85)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Success Rate (prag_margin > 0)")
    ax.set_title("Discourse vs Bare: Success Rate by Model Scale")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.30, color="gray", linestyle="--", linewidth=0.8, label="30% gate")
    ax.legend()

    # ── subplot 2: mean prag_margin ± std ─────────────────────────────────────
    ax = axes[1]
    for i, tname in enumerate(("discourse", "bare")):
        means = [s["stats"][tname]["mean_prag_margin"] for s in all_stats]
        stds  = [s["stats"][tname]["std_prag_margin"]  for s in all_stats]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=tname.capitalize(), color=colors[tname], alpha=0.85)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Mean prag_margin (± std)")
    ax.set_title("Discourse vs Bare: Mean prag_margin by Model Scale")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()

    fig.suptitle(
        "Stage 1b: Discourse Template Advantage Across Pythia Model Scales\n"
        "(prag_margin = score(H_prag) − max(score(H_impl), score(H_lit)))",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stage 1b: Multi-Model Behavioral Pre-screening")
    print("=" * 60)

    # Load items from existing 1.4B prescreen (has premises + hypotheses)
    with open(PRESCREEN_1B4) as f:
        prescreen_data = json.load(f)
    # Convert to common format {premise, hypotheses: {prag, impl, lit}}
    items = [
        {"premise": d["premise"], "hypotheses": d["hypotheses"]}
        for d in prescreen_data
    ]
    print(f"Loaded {len(items)} items from Stage 1 prescreen.\n")

    all_results_stats = []

    # ── run new models ────────────────────────────────────────────────────────
    for model_tag, model_name in MODELS:
        if model_tag == "2b8" and SKIP_2B8:
            print(f"Skipping {model_name} (SKIP_2B8=True).")
            continue

        print(f"\n{'─'*50}")
        print(f"  Loading {model_name}  ({model_tag.upper()})")
        print(f"{'─'*50}")

        try:
            model  = load_model(model_name)
            result = run_model_prescreen(model, items, model_tag)
            unload_model(model)

            # save per-model results
            out_path = SAVE_DIR / f"results_{model_tag}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Results saved → {out_path}")

            all_results_stats.append({"model": model_tag, "stats": result["stats"]})

        except Exception as e:
            print(f"ERROR running {model_name}: {e}", file=sys.stderr)
            unload_model(model) if 'model' in dir() else None

    # ── add 1.4B baseline stats ───────────────────────────────────────────────
    stats_1b4 = extract_1b4_stats(PRESCREEN_1B4)

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY: Discourse vs Bare across Model Scales")
    print("=" * 60)
    # Order: 410M, 1.4B, 2.8B
    ordered_stats = []
    order = ["410m", "1b4", "2b8"]
    stats_map = {s["model"]: s for s in all_results_stats}
    stats_map["1b4"] = stats_1b4

    for tag in order:
        if tag in stats_map:
            ordered_stats.append(stats_map[tag])

    model_labels = {"410m": "Pythia-410M", "1b4": "Pythia-1.4B", "2b8": "Pythia-2.8B"}
    header = f"{'Model':15s}  {'disc_success':>12s}  {'bare_success':>12s}  {'disc_margin':>12s}  {'bare_margin':>12s}  {'delta_success':>14s}"
    print(header)
    print("─" * len(header))
    for s in ordered_stats:
        st  = s["stats"]
        tag = s["model"]
        ds  = st["discourse"]["success_rate"]
        bs  = st["bare"]["success_rate"]
        dm  = st["discourse"]["mean_prag_margin"]
        bm  = st["bare"]["mean_prag_margin"]
        print(f"{model_labels.get(tag, tag):15s}  {ds:12.1%}  {bs:12.1%}  {dm:+12.4f}  {bm:+12.4f}  {ds-bs:+14.1%}")

    # ── plot ──────────────────────────────────────────────────────────────────
    if len(ordered_stats) >= 2:
        plot_comparison(ordered_stats, SAVE_DIR / "fig_scaling.png")
    else:
        print("Not enough models to plot. Run at least one new model.")

    print("\nDone.")


if __name__ == "__main__":
    main()
