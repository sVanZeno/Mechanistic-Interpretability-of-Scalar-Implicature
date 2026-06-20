import json
import gc
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer

ROOT        = Path(__file__).parent.parent
PRESCREEN   = ROOT / "stage1_prescreen" / "prescreen_results.json"
RESULTS_OUT = Path(__file__).parent / "probe_results.json"

N_LAYERS = 24
N_FOLDS  = 5
TEMPLATES = [
    ("bare",      lambda p: p + " "),
    ("discourse", lambda p: f"Context: {p}\nConclusion: "),
]


def load_model():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        "pythia-1.4b",
        dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()
    return model


def extract_prefix_resid(model, prefix: str) -> np.ndarray:
    """
    Run model on prefix only, return hook_resid_post at the LAST prefix
    token for each layer. Shape: (n_layers, d_model) float32.
    """
    prefix_ids = model.to_tokens(prefix, prepend_bos=True)   # (1, seq)
    _, cache = model.run_with_cache(
        prefix_ids,
        names_filter=lambda n: n.endswith("hook_resid_post"),
        device="cpu",
    )
    last = prefix_ids.shape[1] - 1
    acts = np.stack(
        [cache[f"blocks.{l}.hook_resid_post"][0, last, :].float().numpy()
         for l in range(N_LAYERS)],
        axis=0,
    )  # (24, 2048)
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return acts


def probe_binary(X: np.ndarray, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    aurocs = []
    for train_idx, val_idx in skf.split(X, y):
        if len(np.unique(y[val_idx])) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val   = scaler.transform(X[val_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y[train_idx])
        prob = clf.predict_proba(X_val)[:, 1]
        aurocs.append(roc_auc_score(y[val_idx], prob))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def run_probing():
    with open(PRESCREEN) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} triplets.")

    model = load_model()

    # ── extraction ─────────────────────────────────────────────────────────
    # acts[tname] = list of (24, 2048) arrays, one per item
    acts: dict[str, list] = {t: [] for t, _ in TEMPLATES}

    # labels[tname] = dict of label_name -> list of 0/1 per item
    # derived from Stage 1 scores after extraction loop
    scores: dict[str, list[dict]] = {t: [] for t, _ in TEMPLATES}

    for idx, item in enumerate(items):
        premise = item["premise"]
        conds   = item["conditions"]      # {tname: {prag, impl, lit}}

        for tname, prefix_fn in TEMPLATES:
            prefix = prefix_fn(premise)
            acts[tname].append(extract_prefix_resid(model, prefix))
            scores[tname].append(conds[tname])  # {prag, impl, lit}

        if (idx + 1) % 25 == 0:
            print(f"  Extracted {idx + 1}/{len(items)} items")

    print("Extraction done. Computing labels + training probes...")

    probe_results: dict = {}

    for tname, _ in TEMPLATES:
        X = np.stack(acts[tname], axis=0)   # (100, 24, 2048)
        sc = scores[tname]                  # list of {prag, impl, lit}

        # ── labels ─────────────────────────────────────────────────────────
        strict   = np.array([1 if s["prag"] > s["impl"] > s["lit"] else 0 for s in sc])
        prag_win = np.array([1 if s["prag"] >= max(s["impl"], s["lit"]) else 0 for s in sc])
        prag_gt_impl = np.array([1 if s["prag"] > s["impl"] else 0 for s in sc])

        label_sets = {
            "strict_order":   strict,
            "prag_wins":      prag_win,
            "prag_gt_impl":   prag_gt_impl,
        }

        probe_results[tname] = {}
        for label_name, y in label_sets.items():
            pos = int(y.sum())
            print(f"  {tname}/{label_name}: {pos} pos / {len(y)-pos} neg")
            probe_results[tname][label_name] = {}
            for layer in range(N_LAYERS):
                Xi = X[:, layer, :]   # (100, 2048)
                auroc = probe_binary(Xi, y)
                probe_results[tname][label_name][layer] = round(auroc, 4)

    # ── print summary table ────────────────────────────────────────────────
    print("\n=== Probe AUROC (prefix-only, last-token position) ===")
    for tname, _ in TEMPLATES:
        print(f"\n[{tname}]")
        label_names = list(probe_results[tname].keys())
        header = f"  {'layer':>5}  " + "  ".join(f"{n:>15}" for n in label_names)
        print(header)
        for layer in range(N_LAYERS):
            vals = "  ".join(
                f"{probe_results[tname][n][layer]:>15.4f}" for n in label_names
            )
            print(f"  {layer:>5}  {vals}")

    # ── save ───────────────────────────────────────────────────────────────
    with open(RESULTS_OUT, "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"\nResults saved -> {RESULTS_OUT}")


if __name__ == "__main__":
    run_probing()
