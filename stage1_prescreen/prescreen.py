import torch
import torch.nn.functional as F
import json
import gc
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from transformer_lens import HookedTransformer

SAVE_DIR    = Path(__file__).parent
RESULTS_OUT = SAVE_DIR / "prescreen_results.json"
GATE        = 0.30
N_MAX       = 400

TEMPLATES = [
    ("bare",      lambda p: p + " "),
    ("nli",       lambda p: f"Premise: {p}\nHypothesis: "),
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


def score_conditional(model, prefix: str, hypothesis: str) -> float:
    prefix_ids = model.to_tokens(prefix, prepend_bos=True)
    hyp_ids    = model.to_tokens(hypothesis, prepend_bos=False)
    full_ids   = torch.cat([prefix_ids, hyp_ids], dim=1)
    logits     = model(full_ids)
    p_len      = prefix_ids.shape[1]
    h_len      = hyp_ids.shape[1]
    logits_hyp = logits[0, p_len - 1 : p_len + h_len - 1, :]
    log_probs  = F.log_softmax(logits_hyp, dim=-1)
    token_lp   = log_probs[torch.arange(h_len), hyp_ids[0]]
    return (token_lp.sum() / h_len).item()


def load_triplets():
    ds = load_dataset("facebook/imppres", "implicature_quantifiers", split="quantifiers")

    print(f"Dataset size : {len(ds)}")
    print(f"Columns      : {ds.column_names}")
    print(f"\nSample item  :")
    print(json.dumps({k: str(ds[0][k]) for k in ds.column_names}, indent=2))

    from collections import Counter
    rel_counts = Counter((item["spec_relation"], item["item_type"]) for item in ds)
    print("\nAll spec_relation × item_type combinations:")
    for (rel, itype), n in sorted(rel_counts.items()):
        print(f"  {n:4d}  [{itype}]  {rel}")

    # gold labels are encoded as integers: 0=entailment, 1=neutral, 2=contradiction
    prag_rel = "implicature_PtoN"
    impl_rel = "negated implicature_P"

    groups = defaultdict(dict)
    for item in ds:
        premise = item["premise"]
        s_rel   = item["spec_relation"]
        log_l   = int(item["gold_label_log"])
        prag_l  = int(item["gold_label_prag"])
        itype   = item["item_type"]

        if itype == "target" and s_rel == prag_rel:
            groups[premise]["prag"] = item
        elif itype == "target" and s_rel == impl_rel:
            groups[premise]["impl"] = item
        elif log_l == 2 and prag_l == 2 and "lit" not in groups[premise]:
            groups[premise]["lit"] = item

    triplets = [
        v for v in groups.values()
        if all(k in v for k in ("prag", "impl", "lit"))
    ]
    print(f"\nComplete triplets found: {len(triplets)}")
    if triplets:
        t = triplets[0]
        print(f"\nExample premise : {t['prag']['premise']}")
        print(f"  H_prag ({t['prag']['spec_relation']}): {t['prag']['hypothesis']}")
        print(f"  H_impl ({t['impl']['spec_relation']}): {t['impl']['hypothesis']}")
        print(f"  H_lit  ({t['lit']['spec_relation']}): {t['lit']['hypothesis']}")
    return triplets[:N_MAX]


def run_prescreen():
    triplets = load_triplets()
    model    = load_model()
    results  = []

    for idx, trip in enumerate(triplets):
        premise = trip["prag"]["premise"]
        hyps    = {
            "prag": trip["prag"]["hypothesis"],
            "impl": trip["impl"]["hypothesis"],
            "lit":  trip["lit"]["hypothesis"],
        }
        item_result = {"premise": premise, "hypotheses": hyps, "conditions": {}}

        for tname, prefix_fn in TEMPLATES:
            prefix = prefix_fn(premise)
            scores = {k: score_conditional(model, prefix, h) for k, h in hyps.items()}
            item_result["conditions"][tname] = scores

        results.append(item_result)

        if (idx + 1) % 50 == 0:
            print(f"Scored {idx + 1}/{len(triplets)} items")
            gc.collect()
            torch.cuda.empty_cache()

    analyze(results)

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_OUT}")


def analyze(results):
    print("\n=== Pre-screen Analysis ===")
    any_pass = False
    for tname, _ in TEMPLATES:
        ordered = sum(
            1 for r in results
            if (c := r["conditions"].get(tname))
            and c["prag"] > c["impl"] > c["lit"]
        )
        total   = len(results)
        pct     = ordered / total if total else 0
        status  = "PASS" if pct >= GATE else "FAIL"
        if pct >= GATE:
            any_pass = True
        print(f"  {tname:12s}: {ordered}/{total} = {pct:.1%}  {status}")

    print()
    if any_pass:
        print("GO: ≥30% ordering achieved under ≥1 condition. Proceed to Stage 2.")
    else:
        print("NO-GO: <30% under all conditions.")
        print("Action: review prompt design, then fall back to Pythia-2.8B on Colab.")


if __name__ == "__main__":
    run_prescreen()
