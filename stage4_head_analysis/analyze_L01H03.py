"""
Stage 4 — L01H03 Deep Analysis
Experiments:
  A. Attention pattern analysis (what tokens does L01H03 attend to?)
  B. Direct Logit Attribution (what does L01H03 write to logits?)
  C. OV circuit vocabulary projection (weight-level semantic bias)
  D. Mean ablation (causal effect robustness check)
"""

import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
PRESCREEN   = ROOT / "stage1_prescreen" / "prescreen_results.json"
SAVE_DIR    = Path(__file__).parent
RESULTS_OUT = SAVE_DIR / "stage4_results.json"

# ── constants ──────────────────────────────────────────────────────────────
TARGET_LAYER = 1
TARGET_HEAD  = 3
N_HEADS      = 16
D_HEAD       = 128
MIN_DENOM    = 0.3

TEMPLATE_SRC = ("discourse", lambda p: f"Context: {p}\nConclusion: ")
TEMPLATE_TGT = ("bare",      lambda p: p + " ")

# Interesting tokens for OV circuit analysis
PROBE_WORDS = ["some", "all", "not", "none", "no", "most", "few",
               "always", "never", "every", "any", "only"]


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


def score(model, prefix_ids, hyp_ids):
    full_ids = torch.cat([prefix_ids, hyp_ids], dim=1)
    p_len, h_len = prefix_ids.shape[1], hyp_ids.shape[1]
    logits = model(full_ids)
    lp = F.log_softmax(logits[0, p_len - 1 : p_len + h_len - 1], dim=-1)
    return (lp[torch.arange(h_len), hyp_ids[0]].sum() / h_len).item()


def prag_margin(model, prefix_ids, hyps_ids):
    sc = {k: score(model, prefix_ids, hyps_ids[k]) for k in hyps_ids}
    return sc["prag"] - max(sc["impl"], sc["lit"]), sc


def find_some_position(model, prefix_ids):
    """Return the token position of ' some' or 'some' in the prefix."""
    # Try common tokenizations of "some"
    for candidate in [" some", "some", "Some", " Some"]:
        try:
            tok_id = model.to_single_token(candidate)
            ids_list = prefix_ids[0].tolist()
            for pos, tid in enumerate(ids_list):
                if tid == tok_id:
                    return pos
        except Exception:
            continue
    # Fallback: search by string
    for pos in range(prefix_ids.shape[1]):
        tok_str = model.to_string(prefix_ids[0, pos].unsqueeze(0)).strip().lower()
        if tok_str == "some":
            return pos
    return None  # not found


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Attention Pattern
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_A(model, items):
    print_section("Experiment A: Attention Pattern")

    # For each item, record how much attention the last prefix token
    # (query) directs to: [BOS, "some", last_pos, other]
    bos_attns  = []
    some_attns = []
    self_attns = []
    other_attns = []
    some_found  = 0
    raw_patterns = []  # for heatmap of a representative item

    for idx, item in enumerate(items):
        premise = item["premise"]
        prefix  = TEMPLATE_SRC[1](premise)
        prefix_ids = model.to_tokens(prefix, prepend_bos=True)
        last_pos   = prefix_ids.shape[1] - 1

        _, cache = model.run_with_cache(
            prefix_ids,
            names_filter=lambda n: n == f"blocks.{TARGET_LAYER}.attn.hook_pattern",
            device="cpu",
        )
        # pattern shape: (1, n_heads, q_pos, k_pos)
        pattern = cache[f"blocks.{TARGET_LAYER}.attn.hook_pattern"][0, TARGET_HEAD].float().numpy()
        # query = last prefix token; shape: (k_pos,)
        attn = pattern[last_pos]   # (seq_len,)

        # Record raw pattern for heatmap (first 5 items)
        if idx < 5:
            raw_patterns.append(attn)

        bos_attns.append(float(attn[0]))
        self_attns.append(float(attn[last_pos]))

        some_pos = find_some_position(model, prefix_ids)
        if some_pos is not None and some_pos != 0 and some_pos != last_pos:
            some_attns.append(float(attn[some_pos]))
            some_found += 1
            other_mask = np.ones(len(attn), dtype=bool)
            other_mask[0] = False
            other_mask[some_pos] = False
            other_mask[last_pos] = False
            other_attns.append(float(attn[other_mask].sum()))
        else:
            some_attns.append(float("nan"))
            other_mask = np.ones(len(attn), dtype=bool)
            other_mask[0] = False
            other_mask[last_pos] = False
            other_attns.append(float(attn[other_mask].sum()))

        del cache
        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 25 == 0:
            print(f"  A: processed {idx+1}/{len(items)}")

    mean_bos  = float(np.mean(bos_attns))
    mean_some = float(np.nanmean(some_attns))
    mean_self = float(np.mean(self_attns))
    mean_other = float(np.mean(other_attns))

    print(f"  Items with 'some' found: {some_found}/{len(items)}")
    print(f"  Mean attn → BOS   : {mean_bos:.4f}")
    print(f"  Mean attn → 'some': {mean_some:.4f}")
    print(f"  Mean attn → self  : {mean_self:.4f}")
    print(f"  Mean attn → other : {mean_other:.4f}")

    # ── plot: bar chart of attention by category ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["BOS", "'some'", "self (last)", "other"]
    means = [mean_bos, mean_some, mean_self, mean_other]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(categories, means, color=colors, width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(f"L{TARGET_LAYER:02d}H{TARGET_HEAD:02d} — Attention Distribution\n"
                 f"(query = last prefix token, discourse template, n={len(items)})")
    ax.set_ylim(0, max(means) * 1.25)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "figA_attention_pattern.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figA_attention_pattern.png")

    return {
        "mean_attn_bos":   round(mean_bos, 4),
        "mean_attn_some":  round(mean_some, 4),
        "mean_attn_self":  round(mean_self, 4),
        "mean_attn_other": round(mean_other, 4),
        "some_found":      some_found,
        "n_items":         len(items),
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Direct Logit Attribution (DLA)
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_B(model, items):
    print_section("Experiment B: Direct Logit Attribution")

    W_O = model.W_O[TARGET_LAYER, TARGET_HEAD].float().cpu()  # (d_head, d_model)
    W_U = model.W_U.float().cpu()                              # (d_model, d_vocab)

    logit_contributions = []  # accumulate per-item DLA vectors (d_vocab,)

    for idx, item in enumerate(items):
        premise = item["premise"]
        prefix  = TEMPLATE_SRC[1](premise)
        prefix_ids = model.to_tokens(prefix, prepend_bos=True)
        last_pos   = prefix_ids.shape[1] - 1

        _, cache = model.run_with_cache(
            prefix_ids,
            names_filter=lambda n: n == f"blocks.{TARGET_LAYER}.attn.hook_z",
            device="cpu",
        )
        # hook_z shape: (1, seq, n_heads, d_head)
        z = cache[f"blocks.{TARGET_LAYER}.attn.hook_z"][0, last_pos, TARGET_HEAD].float()  # (d_head,)

        # DLA: project through W_O then W_U
        resid_contrib = z @ W_O          # (d_model,)
        logit_contrib = resid_contrib @ W_U  # (d_vocab,)
        logit_contributions.append(logit_contrib.numpy())

        del cache, z, resid_contrib, logit_contrib
        gc.collect()

        if (idx + 1) % 25 == 0:
            print(f"  B: processed {idx+1}/{len(items)}")

    mean_dla = np.mean(logit_contributions, axis=0)  # (d_vocab,)

    # Top 20 positive and negative tokens
    top_pos_idx = np.argsort(mean_dla)[-20:][::-1]
    top_neg_idx = np.argsort(mean_dla)[:20]

    top_pos = [(model.tokenizer.decode([i]).strip(), round(float(mean_dla[i]), 4))
               for i in top_pos_idx]
    top_neg = [(model.tokenizer.decode([i]).strip(), round(float(mean_dla[i]), 4))
               for i in top_neg_idx]

    print("  Top 10 PROMOTED tokens:")
    for tok, val in top_pos[:10]:
        print(f"    {repr(tok):20s}  {val:+.4f}")
    print("  Top 10 SUPPRESSED tokens:")
    for tok, val in top_neg[:10]:
        print(f"    {repr(tok):20s}  {val:+.4f}")

    # ── plot ────────────────────────────────────────────────────────────────
    n_show = 20
    pos_toks = [t for t, _ in top_pos[:n_show]]
    pos_vals = [v for _, v in top_pos[:n_show]]
    neg_toks = [t for t, _ in top_neg[:n_show]]
    neg_vals = [v for _, v in top_neg[:n_show]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(pos_toks[::-1], pos_vals[::-1], color="#55A868")
    ax1.set_xlabel("Mean Logit Contribution")
    ax1.set_title(f"L{TARGET_LAYER:02d}H{TARGET_HEAD:02d} DLA — Top Promoted Tokens")
    ax1.axvline(0, color="black", linewidth=0.8)

    ax2.barh(neg_toks, neg_vals, color="#C44E52")
    ax2.set_xlabel("Mean Logit Contribution")
    ax2.set_title(f"L{TARGET_LAYER:02d}H{TARGET_HEAD:02d} DLA — Top Suppressed Tokens")
    ax2.axvline(0, color="black", linewidth=0.8)

    fig.suptitle("Direct Logit Attribution — L01H03\n"
                 f"(averaged over {len(items)} discourse-template items)", fontsize=11)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "figB_DLA.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figB_DLA.png")

    # Check specific pragmatically relevant tokens
    pragma_check = {}
    for word in ["not", "all", "some", "none", "no", "most", "few", "every", "any"]:
        try:
            for candidate in [" " + word, word]:
                try:
                    tid = model.to_single_token(candidate)
                    pragma_check[word] = round(float(mean_dla[tid]), 4)
                    break
                except Exception:
                    continue
        except Exception:
            pragma_check[word] = None
    print("  DLA for pragmatic tokens:")
    for w, v in pragma_check.items():
        print(f"    {w:10s}: {v}")

    return {
        "top_promoted":  top_pos[:20],
        "top_suppressed": top_neg[:20],
        "pragmatic_tokens_dla": pragma_check,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — OV Circuit Vocabulary Projection
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_C(model):
    print_section("Experiment C: OV Circuit Vocabulary Projection")

    W_V = model.W_V[TARGET_LAYER, TARGET_HEAD].float().cpu()  # (d_model, d_head)
    W_O = model.W_O[TARGET_LAYER, TARGET_HEAD].float().cpu()  # (d_head, d_model)
    W_E = model.W_E.float().cpu()                              # (d_vocab, d_model)
    W_U = model.W_U.float().cpu()                              # (d_model, d_vocab)

    # W_OV = W_V @ W_O: (d_model, d_model)
    W_OV = W_V @ W_O

    results = {}
    all_effects = []  # (n_probe_words, d_vocab) for heatmap

    print("  OV effect: if L01H03 attends to token X, what logits are promoted?")
    print(f"  {'Token':15s}  Top promoted         Top suppressed")

    for word in PROBE_WORDS:
        tok_id = None
        for candidate in [" " + word, word, " " + word.capitalize()]:
            try:
                tok_id = model.to_single_token(candidate)
                break
            except Exception:
                continue
        if tok_id is None:
            results[word] = None
            continue

        e = W_E[tok_id]             # (d_model,)
        ov_out = e @ W_OV           # (d_model,)
        logit_shift = ov_out @ W_U  # (d_vocab,)
        logit_shift_np = logit_shift.numpy()
        all_effects.append(logit_shift_np)

        top_p_idx = np.argsort(logit_shift_np)[-5:][::-1]
        top_n_idx = np.argsort(logit_shift_np)[:5]
        top_p = [model.tokenizer.decode([i]).strip() for i in top_p_idx]
        top_n = [model.tokenizer.decode([i]).strip() for i in top_n_idx]
        top_p_vals = [round(float(logit_shift_np[i]), 3) for i in top_p_idx]
        top_n_vals = [round(float(logit_shift_np[i]), 3) for i in top_n_idx]

        print(f"  {repr(word):15s}  {top_p[0]}({top_p_vals[0]:+.2f}) {top_p[1]}({top_p_vals[1]:+.2f})   "
              f"{top_n[0]}({top_n_vals[0]:+.2f}) {top_n[1]}({top_n_vals[1]:+.2f})")

        results[word] = {
            "tok_id":       tok_id,
            "top_promoted": list(zip(top_p, top_p_vals)),
            "top_suppressed": list(zip(top_n, top_n_vals)),
        }

    # ── plot: heatmap of OV effects for probe words ─────────────────────────
    # Show effect on a set of "output" tokens of interest
    output_words = ["not", "all", "some", "no", "most", "few",
                    "every", "any", "only", "always", "never", "both"]
    valid_probe = [w for w in PROBE_WORDS if results.get(w) is not None]

    output_ids = []
    output_labels = []
    for w in output_words:
        for candidate in [" " + w, w]:
            try:
                tid = model.to_single_token(candidate)
                output_ids.append(tid)
                output_labels.append(repr(candidate))
                break
            except Exception:
                continue

    if all_effects and output_ids:
        matrix = np.stack(all_effects, axis=0)[:, output_ids]  # (n_probe, n_output)
        fig, ax = plt.subplots(figsize=(max(8, len(output_ids)), max(4, len(valid_probe))))
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(output_labels)))
        ax.set_xticklabels(output_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(valid_probe)))
        ax.set_yticklabels([repr(w) for w in valid_probe], fontsize=9)
        plt.colorbar(im, ax=ax, label="Logit shift")
        ax.set_title(f"L{TARGET_LAYER:02d}H{TARGET_HEAD:02d} OV Circuit\n"
                     f"Row = input token attended to, Col = output logit promoted/suppressed")
        fig.tight_layout()
        fig.savefig(SAVE_DIR / "figC_OV_circuit.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: figC_OV_circuit.png")

    return results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — Mean Ablation
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_D(model, items):
    print_section("Experiment D: Mean Ablation")

    # Step 1: collect mean hook_z for L01H03 across all items (discourse template)
    print("  D1: Computing mean hook_z across all items...")
    all_z = []
    for idx, item in enumerate(items):
        premise = item["premise"]
        prefix  = TEMPLATE_SRC[1](premise)
        prefix_ids = model.to_tokens(prefix, prepend_bos=True)
        last_pos   = prefix_ids.shape[1] - 1

        _, cache = model.run_with_cache(
            prefix_ids,
            names_filter=lambda n: n == f"blocks.{TARGET_LAYER}.attn.hook_z",
            device="cpu",
        )
        z = cache[f"blocks.{TARGET_LAYER}.attn.hook_z"][0, last_pos, TARGET_HEAD].float()
        all_z.append(z.numpy())
        del cache
        gc.collect()

    mean_z = torch.tensor(np.mean(all_z, axis=0), dtype=torch.float32).to("cuda")  # (d_head,)
    print(f"  Mean z computed over {len(all_z)} items, shape: {mean_z.shape}")

    # Step 2: for each bare-template item, compute prag_margin
    # baseline (no ablation) vs mean-ablated vs zero-ablated
    print("  D2: Running ablation on bare-template items...")

    results_abl = []
    for idx, item in enumerate(items):
        premise = item["premise"]
        hyps    = item["hypotheses"]
        tgt_ids = model.to_tokens(TEMPLATE_TGT[1](premise), prepend_bos=True)
        tgt_pos = tgt_ids.shape[1] - 1
        hyp_ids = {k: model.to_tokens(v, prepend_bos=False) for k, v in hyps.items()}

        # baseline
        sc_base = {k: score(model, tgt_ids, hyp_ids[k]) for k in hyps}
        m_base  = sc_base["prag"] - max(sc_base["impl"], sc_base["lit"])

        # mean ablation
        def make_mean_hook(pos=tgt_pos, head=TARGET_HEAD, mz=mean_z):
            def _fn(val, hook):
                val[:, pos, head, :] = mz
                return val
            return _fn

        sc_mean = {}
        for k in hyps:
            full_ids = torch.cat([tgt_ids, hyp_ids[k]], dim=1)
            p_len, h_len = tgt_ids.shape[1], hyp_ids[k].shape[1]
            logits = model.run_with_hooks(
                full_ids,
                fwd_hooks=[(f"blocks.{TARGET_LAYER}.attn.hook_z", make_mean_hook())]
            )
            lp = F.log_softmax(logits[0, p_len - 1: p_len + h_len - 1], dim=-1)
            sc_mean[k] = (lp[torch.arange(h_len), hyp_ids[k][0]].sum() / h_len).item()
        m_mean = sc_mean["prag"] - max(sc_mean["impl"], sc_mean["lit"])

        # zero ablation
        def make_zero_hook(pos=tgt_pos, head=TARGET_HEAD):
            def _fn(val, hook):
                val[:, pos, head, :] = 0.0
                return val
            return _fn

        sc_zero = {}
        for k in hyps:
            full_ids = torch.cat([tgt_ids, hyp_ids[k]], dim=1)
            p_len, h_len = tgt_ids.shape[1], hyp_ids[k].shape[1]
            logits = model.run_with_hooks(
                full_ids,
                fwd_hooks=[(f"blocks.{TARGET_LAYER}.attn.hook_z", make_zero_hook())]
            )
            lp = F.log_softmax(logits[0, p_len - 1: p_len + h_len - 1], dim=-1)
            sc_zero[k] = (lp[torch.arange(h_len), hyp_ids[k][0]].sum() / h_len).item()
        m_zero = sc_zero["prag"] - max(sc_zero["impl"], sc_zero["lit"])

        results_abl.append({
            "baseline": m_base,
            "mean_abl": m_mean,
            "zero_abl": m_zero,
        })

        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 25 == 0:
            print(f"  D: processed {idx+1}/{len(items)}")

    base_margins = np.array([r["baseline"] for r in results_abl])
    mean_margins = np.array([r["mean_abl"] for r in results_abl])
    zero_margins = np.array([r["zero_abl"] for r in results_abl])

    delta_mean = float(np.mean(mean_margins - base_margins))
    delta_zero = float(np.mean(zero_margins - base_margins))

    print(f"  Mean ablation effect on prag_margin: {delta_mean:+.4f}")
    print(f"  Zero ablation effect on prag_margin: {delta_zero:+.4f}")
    print(f"  (Negative = removing head hurts pragmatic preference)")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    conditions = ["Baseline\n(no ablation)", "Mean ablation", "Zero ablation"]
    means_d = [float(np.mean(base_margins)),
               float(np.mean(mean_margins)),
               float(np.mean(zero_margins))]
    stds_d  = [float(np.std(base_margins)),
               float(np.std(mean_margins)),
               float(np.std(zero_margins))]
    colors_d = ["#4C72B0", "#DD8452", "#C44E52"]
    bars = ax.bar(conditions, means_d, yerr=stds_d, capsize=5,
                  color=colors_d, width=0.5, alpha=0.85)
    for bar, val in zip(bars, means_d):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds_d) * 0.05,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean prag_margin")
    ax.set_title(f"L{TARGET_LAYER:02d}H{TARGET_HEAD:02d} — Ablation Effect on prag_margin\n"
                 f"(bare template, n={len(items)})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "figD_ablation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: figD_ablation.png")

    return {
        "mean_baseline":     round(float(np.mean(base_margins)), 4),
        "mean_after_mean_abl": round(float(np.mean(mean_margins)), 4),
        "mean_after_zero_abl": round(float(np.mean(zero_margins)), 4),
        "delta_mean_abl":    round(delta_mean, 4),
        "delta_zero_abl":    round(delta_zero, 4),
        "n_items":           len(items),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    with open(PRESCREEN) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from prescreen.")

    model = load_model()
    print("Model loaded.")

    results = {}

    # ── A ──────────────────────────────────────────────────────────────────
    results["A_attention_pattern"] = run_experiment_A(model, items)

    # ── B ──────────────────────────────────────────────────────────────────
    results["B_DLA"] = run_experiment_B(model, items)

    # ── C ──────────────────────────────────────────────────────────────────
    results["C_OV_circuit"] = run_experiment_C(model)

    # ── D ──────────────────────────────────────────────────────────────────
    results["D_mean_ablation"] = run_experiment_D(model, items)

    # ── save ───────────────────────────────────────────────────────────────
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved -> {RESULTS_OUT}")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 4 SUMMARY")
    print("="*60)
    A = results["A_attention_pattern"]
    print(f"  A. Attention:  BOS={A['mean_attn_bos']:.3f}  'some'={A['mean_attn_some']:.3f}"
          f"  self={A['mean_attn_self']:.3f}  other={A['mean_attn_other']:.3f}")
    D = results["D_mean_ablation"]
    print(f"  D. Mean abl:   prag_margin baseline={D['mean_baseline']:.4f}"
          f"  after mean-abl={D['mean_after_mean_abl']:.4f}"
          f"  (delta={D['delta_mean_abl']:+.4f})")
    print("\nFigures: figA_attention_pattern.png  figB_DLA.png  figC_OV_circuit.png  figD_ablation.png")


if __name__ == "__main__":
    main()
