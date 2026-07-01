"""
Stage 4e — L01H03 Attention Breakdown by Token Category
==========================================================
目的：细分 Stage 4A 中 "other" 桶（占 86.6%）的内部组成，
      并对比 discourse vs. bare 两种模板下 L01H03 的注意力分布。

Token 分类（discourse 模板）：
  bos      - position 0 (BOS token)
  struct   - "Context: " 前缀 + "\nConclusion: " 后缀的所有 token
  quant    - "some" token 所在位置
  content  - premise 中除 "some" 之外的其余词
  self     - last prefix token（query 自身）

Token 分类（bare 模板）：
  bos      - position 0
  quant    - "some" token
  content  - premise 中其余 token + 末尾空格之前的内容
  self     - last token（末尾空格）
  struct   - (不存在，为 0)

科学假设：
  若 L01H03 是"话语结构头"：
    discourse 中 struct 权重 >> bare 中 struct 权重（=0），
    且 discourse 的 struct 权重明显高于 content 权重。
  若 L01H03 是"内容注意力头"：
    两种模板下 content 权重均占主导，struct 权重次要。
"""

import json
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
PRESCREEN   = ROOT / "stage1_prescreen" / "prescreen_results.json"
SAVE_DIR    = Path(__file__).parent
RESULTS_OUT = SAVE_DIR / "stage4e_results.json"

TARGET_LAYER = 1
TARGET_HEAD  = 3

TEMPLATE_DISCOURSE = ("discourse", lambda p: f"Context: {p}\nConclusion: ")
TEMPLATE_BARE      = ("bare",      lambda p: p + " ")

CATEGORIES = ["bos", "struct", "quant", "content", "self"]
COLORS = {
    "bos":     "#8172B2",
    "struct":  "#DD8452",   # orange — the key category
    "quant":   "#55A868",   # green  — "some"
    "content": "#4C72B0",   # blue
    "self":    "#C44E52",   # red
}


# ── helpers ────────────────────────────────────────────────────────────────

def load_model():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        "pythia-1.4b",
        dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()
    return model


def find_token_id(model, word):
    """Try common surface forms of a word; return token id or None."""
    for candidate in [" " + word, word, " " + word.capitalize(), word.capitalize()]:
        try:
            return model.to_single_token(candidate)
        except Exception:
            continue
    return None


def classify_positions(model, prefix_ids, template_type):
    """
    Return dict {category: [list_of_positions]} for a single prefix.
    All positions are indices into prefix_ids[0].
    """
    ids_list  = prefix_ids[0].tolist()
    seq_len   = len(ids_list)
    last_pos  = seq_len - 1
    assigned  = set()
    cats      = {c: [] for c in CATEGORIES}

    # BOS
    cats["bos"].append(0)
    assigned.add(0)

    # Self (last token)
    cats["self"].append(last_pos)
    assigned.add(last_pos)

    if template_type == "discourse":
        # --- structural prefix: tokens for "Context: " (no BOS) ---
        struct_prefix_ids = model.to_tokens("Context: ", prepend_bos=False)[0].tolist()
        n_sp = len(struct_prefix_ids)
        # These should occupy positions 1 … n_sp  (right after BOS)
        for i in range(1, 1 + n_sp):
            if i < seq_len and ids_list[i] == struct_prefix_ids[i - 1]:
                cats["struct"].append(i)
                assigned.add(i)

        # --- structural suffix: find '\n' and classify up to last_pos-1 ---
        # '\n' tokenizes as a single token in Pythia (NeoX tokenizer)
        newline_id = None
        for candidate in ["\n", "Ċ"]:   # GPT-NeoX uses Ċ for \n
            try:
                newline_id = model.to_single_token(candidate)
                break
            except Exception:
                continue
        if newline_id is not None:
            for pos in range(1, last_pos):
                if ids_list[pos] == newline_id and pos not in assigned:
                    # From this position to last_pos-1: structural suffix
                    for j in range(pos, last_pos):
                        cats["struct"].append(j)
                        assigned.add(j)
                    break

    # --- quantifier: find "some" ---
    some_id = find_token_id(model, "some")
    if some_id is not None:
        for pos in range(1, last_pos):
            if ids_list[pos] == some_id and pos not in assigned:
                cats["quant"].append(pos)
                assigned.add(pos)
                break  # take first occurrence

    # --- content: everything remaining ---
    for pos in range(seq_len):
        if pos not in assigned:
            cats["content"].append(pos)
            assigned.add(pos)

    return cats


def extract_attention_by_category(model, items, template_name, template_fn):
    """
    For each item, extract L01H03's attention from the last prefix token (query)
    to all key positions, then sum by category.
    Returns per-item array of shape (n_items, len(CATEGORIES)).
    """
    hook_name = f"blocks.{TARGET_LAYER}.attn.hook_pattern"
    results   = []

    for idx, item in enumerate(items):
        premise    = item["premise"]
        prefix     = template_fn(premise)
        prefix_ids = model.to_tokens(prefix, prepend_bos=True)
        last_pos   = prefix_ids.shape[1] - 1

        _, cache = model.run_with_cache(
            prefix_ids,
            names_filter=lambda n: n == hook_name,
            device="cpu",
        )
        # attn shape: (1, n_heads, q_pos, k_pos)
        attn = cache[hook_name][0, TARGET_HEAD, last_pos].float().numpy()  # (k_pos,)

        cats = classify_positions(model, prefix_ids, template_name)

        row = np.zeros(len(CATEGORIES))
        for ci, cat in enumerate(CATEGORIES):
            positions = cats[cat]
            if positions:
                row[ci] = float(attn[positions].sum())

        results.append(row)

        del cache
        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 25 == 0:
            print(f"    [{template_name}] {idx+1}/{len(items)}")

    return np.array(results)   # (n_items, 5)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    with open(PRESCREEN) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items.")

    model = load_model()
    print("Model loaded.\n")

    # ── run for both templates ─────────────────────────────────────────────
    print("=== Discourse template ===")
    arr_disc = extract_attention_by_category(
        model, items, "discourse", TEMPLATE_DISCOURSE[1])

    print("\n=== Bare template ===")
    arr_bare = extract_attention_by_category(
        model, items, "bare", TEMPLATE_BARE[1])

    # ── compute means ─────────────────────────────────────────────────────
    mean_disc = arr_disc.mean(axis=0)
    mean_bare = arr_bare.mean(axis=0)
    std_disc  = arr_disc.std(axis=0)
    std_bare  = arr_bare.std(axis=0)

    print("\n=== Results ===")
    print(f"{'Category':10s}  {'Discourse':>12s}  {'Bare':>12s}  {'Diff (D-B)':>12s}")
    print("-" * 52)
    for ci, cat in enumerate(CATEGORIES):
        d, b = mean_disc[ci], mean_bare[ci]
        print(f"{cat:10s}  {d:12.4f}  {b:12.4f}  {d-b:+12.4f}")

    # ── structured key stats ───────────────────────────────────────────────
    struct_idx  = CATEGORIES.index("struct")
    content_idx = CATEGORIES.index("content")
    quant_idx   = CATEGORIES.index("quant")
    bos_idx     = CATEGORIES.index("bos")

    disc_struct_vs_content = float(mean_disc[struct_idx] / (mean_disc[content_idx] + 1e-9))
    bare_struct            = float(mean_bare[struct_idx])

    print(f"\nDisc struct / content ratio : {disc_struct_vs_content:.3f}")
    print(f"Bare struct weight (expected ~0): {bare_struct:.4f}")
    print(f"\nHypothesis check:")
    if mean_disc[struct_idx] > mean_disc[content_idx]:
        print("  -> discourse struct > content: STRUCTURAL HEAD confirmed")
    else:
        print("  -> discourse content > struct: CONTENT ATTENTION HEAD")

    # ── save results ───────────────────────────────────────────────────────
    results = {
        "categories": CATEGORIES,
        "discourse": {
            "mean": {c: round(float(mean_disc[i]), 4) for i, c in enumerate(CATEGORIES)},
            "std":  {c: round(float(std_disc[i]),  4) for i, c in enumerate(CATEGORIES)},
        },
        "bare": {
            "mean": {c: round(float(mean_bare[i]), 4) for i, c in enumerate(CATEGORIES)},
            "std":  {c: round(float(std_bare[i]),  4) for i, c in enumerate(CATEGORIES)},
        },
        "discourse_struct_vs_content_ratio": round(disc_struct_vs_content, 4),
        "interpretation": (
            "structural_head"
            if mean_disc[struct_idx] > mean_disc[content_idx]
            else "content_attention_head"
        ),
        "n_items": len(items),
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {RESULTS_OUT}")

    # ── plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x    = np.arange(len(CATEGORIES))
    cats_display = ["BOS", "Struct\n(Context/\\n/Concl.)", "'some'", "Content\n(premise words)", "Self\n(last tok)"]

    for ax, (template, mean_arr, std_arr) in zip(
        axes,
        [("Discourse template", mean_disc, std_disc),
         ("Bare template",      mean_bare, std_bare)]
    ):
        bar_colors = [COLORS[c] for c in CATEGORIES]
        bars = ax.bar(x, mean_arr, yerr=std_arr, capsize=4,
                      color=bar_colors, width=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cats_display, fontsize=9)
        ax.set_title(f"L01H03 Attention — {template}\n(n={len(items)})", fontsize=10)
        ax.set_ylabel("Mean attention weight" if ax is axes[0] else "")
        ax.set_ylim(0, max(mean_disc.max(), mean_bare.max()) * 1.30)
        for bar, val in zip(bars, mean_arr):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(
        "Stage 4e: L01H03 Attention by Token Category — Discourse vs. Bare",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out_png = SAVE_DIR / "figE_attn_breakdown.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_png}")

    # ── compact summary for terminal ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4e SUMMARY")
    print("=" * 60)
    for cat, cd, cb in zip(CATEGORIES, mean_disc, mean_bare):
        bar_d = "#" * int(cd * 40)
        bar_b = "#" * int(cb * 40)
        print(f"  {cat:8s}  disc={cd:.3f} {bar_d}")
        print(f"           bare={cb:.3f} {bar_b}")
    print(f"\n  struct/content ratio (discourse): {disc_struct_vs_content:.3f}")
    print(f"  Interpretation: {results['interpretation']}")


if __name__ == "__main__":
    main()
