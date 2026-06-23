import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformer_lens import HookedTransformer

ROOT        = Path(__file__).parent.parent
PRESCREEN   = ROOT / "stage1_prescreen" / "prescreen_results.json"
RESULTS_OUT = Path(__file__).parent / "patch_heads_results.json"

N_HEADS       = 16
TARGET_LAYERS = list(range(1, 6))  # layers 1-5: peak causal effect from stage3b
MIN_DENOM     = 0.3

TEMPLATE_SRC = ("discourse", lambda p: f"Context: {p}\nConclusion: ")
TEMPLATE_TGT = ("bare",      lambda p: p + " ")


def load_model():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        "pythia-1.4b", dtype=torch.bfloat16, device="cuda",
    )
    model.eval()
    return model


def score(model, prefix_ids, hyp_ids, hook_name=None, hook_fn=None):
    full_ids = torch.cat([prefix_ids, hyp_ids], dim=1)
    p_len, h_len = prefix_ids.shape[1], hyp_ids.shape[1]
    if hook_name is not None:
        logits = model.run_with_hooks(full_ids, fwd_hooks=[(hook_name, hook_fn)])
    else:
        logits = model(full_ids)
    lp = F.log_softmax(logits[0, p_len - 1 : p_len + h_len - 1], dim=-1)
    return (lp[torch.arange(h_len), hyp_ids[0]].sum() / h_len).item()


def prag_margin(s):
    return s["prag"] - max(s["impl"], s["lit"])


def run_head_patching():
    with open(PRESCREEN) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items, scanning layers {TARGET_LAYERS}")

    model = load_model()
    target_hook_names = {f"blocks.{l}.attn.hook_z" for l in TARGET_LAYERS}
    head_data = {l: {h: [] for h in range(N_HEADS)} for l in TARGET_LAYERS}

    for idx, item in enumerate(items):
        premise = item["premise"]
        hyps    = item["hypotheses"]

        src_ids = model.to_tokens(TEMPLATE_SRC[1](premise), prepend_bos=True)
        tgt_ids = model.to_tokens(TEMPLATE_TGT[1](premise), prepend_bos=True)
        tgt_pos = tgt_ids.shape[1] - 1

        hyp_ids = {k: model.to_tokens(v, prepend_bos=False) for k, v in hyps.items()}

        _, src_cache = model.run_with_cache(
            src_ids,
            names_filter=lambda n, s=target_hook_names: n in s,
            device="cpu",
        )

        b_src = {k: score(model, src_ids, hyp_ids[k]) for k in hyps}
        b_tgt = {k: score(model, tgt_ids, hyp_ids[k]) for k in hyps}
        m_src, m_tgt = prag_margin(b_src), prag_margin(b_tgt)
        denom = m_src - m_tgt

        for layer in TARGET_LAYERS:
            src_z = src_cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :].to("cuda")

            for head in range(N_HEADS):
                head_val = src_z[head, :]

                def make_hook(hv=head_val, h=head, pos=tgt_pos):
                    def _fn(val, hook):
                        val[:, pos, h, :] = hv
                        return val
                    return _fn

                sc_pat = {
                    k: score(model, tgt_ids, hyp_ids[k],
                             hook_name=f"blocks.{layer}.attn.hook_z",
                             hook_fn=make_hook())
                    for k in hyps
                }
                effect = (prag_margin(sc_pat) - m_tgt) / denom if abs(denom) >= MIN_DENOM else None
                head_data[layer][head].append(effect)

        del src_cache
        gc.collect()
        torch.cuda.empty_cache()

        if (idx + 1) % 20 == 0:
            print(f"  Patched {idx + 1}/{len(items)} items")

    print("\n=== Head patching effect (discourse -> bare, normalized) ===")
    header = f"  {'':6}" + "".join(f"  H{h:02d} " for h in range(N_HEADS))
    print(header)

    results = {}
    top_heads = []
    for layer in TARGET_LAYERS:
        row = f"  L{layer:02d}    "
        results[str(layer)] = {}
        for head in range(N_HEADS):
            valid = [e for e in head_data[layer][head] if e is not None]
            mean  = float(np.mean(valid)) if valid else float("nan")
            results[str(layer)][str(head)] = round(mean, 4)
            top_heads.append((mean, layer, head))
            row += f"  {mean:+.3f}"
        print(row)

    print("\n=== Top 10 heads by mean effect ===")
    top_heads.sort(reverse=True)
    for mean, layer, head in top_heads[:10]:
        print(f"  L{layer:02d}H{head:02d}  {mean:+.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {RESULTS_OUT}")


if __name__ == "__main__":
    run_head_patching()
