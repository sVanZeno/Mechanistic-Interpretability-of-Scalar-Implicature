# Error History — Mistake Notebook

Bugs a first-timer would realistically hit while building this scalar-implicature
mechanistic-interpretability project from scratch. The current code is mostly correct, so
each entry is written backwards: the wrong version first, what broke, then the fix.

Every entry has the same four parts:
- **Where** — which file / function
- **Original code** — the version that fails
- **Error / problem** — what it threw, or the silent wrong result
- **Fix** — the working code

---

## Stage 0 — Environment & model loading (`stage0_environment/verify_env.py`)

### 1. transformer_lens won't import
**Where:** top of every script, `import transformer_lens`.
**Original code:**
```python
# installed with: pip install transformer_lens
import transformer_lens
```
**Error / problem:** installing the normal PyPI `transformer_lens` clashes with the RelP
fork this repo needs, and pip tries to pull a specific old `transformers`, giving
`ERROR: pip's dependency resolver ...` conflicts. Import then fails or loads the wrong version.
**Fix:** install the local RelP fork without letting pip touch dependencies:
```bash
pip install -e ./RelP/TransformerLens --no-deps
```

### 2. `transformers` version conflict breaks model loading
**Where:** `HookedTransformer.from_pretrained(...)`.
**Original code:**
```python
# fresh transformers 4.57.1 in the env
model = HookedTransformer.from_pretrained("pythia-1.4b")
```
**Error / problem:** the RelP fork was written against an older `transformers`; a brand new
one throws `ImportError: cannot import name ... from transformers`.
**Fix:** keep the env's existing torch/transformers (`--system-site-packages` venv) and
install the fork with `--no-deps` so it doesn't upgrade/downgrade anything.

### 3. Model loads in fp32 and blows the 8 GB GPU
**Where:** `load_model()`.
**Original code:**
```python
model = HookedTransformer.from_pretrained("pythia-1.4b", device="cuda")
```
**Error / problem:** default dtype is float32, so 1.4B params want ~5.5 GB just for weights
plus activations → `torch.cuda.OutOfMemoryError` on a 4060 (8 GB).
**Fix:** load in bfloat16:
```python
model = HookedTransformer.from_pretrained("pythia-1.4b", dtype=torch.bfloat16, device="cuda")
```

### 4. Gradients pile up during inference
**Where:** any scoring loop.
**Original code:**
```python
model = HookedTransformer.from_pretrained(...)
# straight into the loop, grad still on
```
**Error / problem:** autograd keeps the graph for every forward pass, VRAM climbs until OOM
even though we never call `.backward()`.
**Fix:** turn grad off globally before the loop:
```python
torch.set_grad_enabled(False)
```

### 5. Forgot `model.eval()`
**Where:** `load_model()`.
**Original code:**
```python
model = HookedTransformer.from_pretrained(...)
return model
```
**Error / problem:** model stays in train mode; dropout is active so the same input gives
slightly different scores each run — results aren't reproducible.
**Fix:**
```python
model.eval()
return model
```

### 6. `get_device_properties(0)` crashes on a CPU-only machine
**Where:** `check_packages()`.
**Original code:**
```python
props = torch.cuda.get_device_properties(0)
print(props.name)
```
**Error / problem:** on a machine with no GPU this raises
`AssertionError: Torch not compiled with CUDA enabled` / index error.
**Fix:** guard it:
```python
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    ...
```

### 7. VRAM never frees between the two test models
**Where:** `smoke_test_160m()` → `vram_test_1_4b()`.
**Original code:**
```python
model = HookedTransformer.from_pretrained("pythia-160m", ...)
# ... run smoke test, then load 1.4b next without cleanup
```
**Error / problem:** the 160M model still sits in VRAM when the 1.4B loads → OOM.
**Fix:** delete and clear before loading the next one:
```python
del model, cache
gc.collect()
torch.cuda.empty_cache()
```

### 8. Reading peak VRAM without resetting the counter
**Where:** `vram_test_1_4b()`.
**Original code:**
```python
model = HookedTransformer.from_pretrained("pythia-1.4b", ...)
peak = torch.cuda.max_memory_allocated()
```
**Error / problem:** `max_memory_allocated` reports the peak since the process started, so it
includes the earlier 160M run — the number is wrong.
**Fix:** reset first:
```python
torch.cuda.reset_peak_memory_stats()
model = HookedTransformer.from_pretrained(...)
```

### 9. `transformer_lens.__version__` doesn't exist on the fork
**Where:** `check_packages()`.
**Original code:**
```python
print(transformer_lens.__version__)
```
**Error / problem:** the RelP fork doesn't set `__version__` → `AttributeError`.
**Fix:** fall back to a default:
```python
tl_ver = getattr(transformer_lens, "__version__", "RelP-fork")
```

### 10. First `from_pretrained` hangs / fails offline
**Where:** first model load.
**Original code:**
```python
model = HookedTransformer.from_pretrained("pythia-1.4b", ...)
```
**Error / problem:** weights download from Hugging Face on first run; with no internet or a
proxy you get `ConnectionError` / `OSError: Can't load ...`.
**Fix:** run once online to cache the weights (or set `HF_HOME`), then it loads from disk.

### 11. Wrong model name string
**Where:** `from_pretrained`.
**Original code:**
```python
model = HookedTransformer.from_pretrained("Pythia-1.4B")
```
**Error / problem:** TransformerLens keys are lowercase with hyphens; `"Pythia-1.4B"` isn't
recognized → `ValueError: ... not found`.
**Fix:** use `"pythia-1.4b"` (also `pythia-410m`, `pythia-2.8b`).

### 12. `datasets` / `transformers` not imported before use in checks
**Where:** `check_packages()`.
**Original code:**
```python
print(datasets.__version__)   # datasets never imported
```
**Error / problem:** `NameError: name 'datasets' is not defined`.
**Fix:** import inside the function: `import transformers, datasets`.

---

## Stage 1 — Behavioral pre-screen (`stage1_prescreen/prescreen.py`)

### 13. Wrong dataset config / split names
**Where:** `load_triplets()`.
**Original code:**
```python
ds = load_dataset("facebook/imppres", "implicature_quantifiers")
for item in ds:   # iterating a DatasetDict
```
**Error / problem:** IMPPRES needs both a config and a split; without `split=...` you get a
`DatasetDict`, and iterating it yields split *names* (strings), so `item["premise"]` throws
`TypeError: string indices must be integers`.
**Fix:**
```python
ds = load_dataset("facebook/imppres", "implicature_quantifiers", split="quantifiers")
```

### 14. Gold labels are strings, not ints
**Where:** `load_triplets()`.
**Original code:**
```python
if item["gold_label_log"] == 2 and item["gold_label_prag"] == 2:
```
**Error / problem:** labels come as `"2"`, so `"2" == 2` is always False → zero triplets
found, empty run, no error.
**Fix:** `int(item["gold_label_log"])`, `int(item["gold_label_prag"])`.

### 15. Same premise grabs multiple "lit" items
**Where:** triplet grouping loop.
**Original code:**
```python
elif log_l == 2 and prag_l == 2:
    groups[premise]["lit"] = item     # overwrites every time
```
**Error / problem:** several rows match the literal condition per premise, so `lit` keeps
getting overwritten with the last one — inconsistent triplets.
**Fix:** only take the first:
```python
elif log_l == 2 and prag_l == 2 and "lit" not in groups[premise]:
    groups[premise]["lit"] = item
```

### 16. Keeping incomplete triplets
**Where:** building the `triplets` list.
**Original code:**
```python
triplets = list(groups.values())
```
**Error / problem:** some premises are missing `prag`, `impl`, or `lit`; later
`trip["prag"]` throws `KeyError`.
**Fix:** filter to complete ones:
```python
triplets = [v for v in groups.values() if all(k in v for k in ("prag","impl","lit"))]
```

### 17. Hypothesis gets its own BOS token
**Where:** `score_conditional()`.
**Original code:**
```python
hyp_ids = model.to_tokens(hypothesis)   # prepend_bos defaults True
full_ids = torch.cat([prefix_ids, hyp_ids], dim=1)
```
**Error / problem:** a `<BOS>` lands in the middle of `prefix + hypothesis`; the model
never sees text like that, so all conditional scores are corrupted.
**Fix:** `hyp_ids = model.to_tokens(hypothesis, prepend_bos=False)`.

### 18. Off-by-one when slicing logits
**Where:** `score_conditional()`.
**Original code:**
```python
logits_hyp = logits[0, p_len : p_len + h_len, :]
```
**Error / problem:** logits at position `t` predict token `t+1`, so the first hypothesis
token's probability lives at `p_len-1`. This slice is shifted one step late → wrong scores,
but it runs fine and hides.
**Fix:** `logits[0, p_len - 1 : p_len + h_len - 1, :]`.

### 19. Gathering the wrong token probabilities
**Where:** `score_conditional()`.
**Original code:**
```python
token_lp = log_probs[:, hyp_ids[0]]   # wrong indexing, picks columns not diagonal
```
**Error / problem:** you want each row's probability for *its* target token (the diagonal),
not every target for every row → shape blows up / nonsense scores.
**Fix:**
```python
token_lp = log_probs[torch.arange(h_len), hyp_ids[0]]
```

### 20. Not normalizing by hypothesis length
**Where:** `score_conditional()`.
**Original code:**
```python
return token_lp.sum().item()
```
**Error / problem:** longer hypotheses get more-negative sums just for being longer, so the
comparison between prag/impl/lit is biased by length, not meaning.
**Fix:** average per token:
```python
return (token_lp.sum() / h_len).item()
```

### 21. Printing ✓/✗ crashes the Windows console
**Where:** `analyze()`.
**Original code:**
```python
status = "✓" if pct >= GATE else "✗"
```
**Error / problem:** Windows GBK console → `UnicodeEncodeError: 'gbk' codec can't encode
character '✓'`, right at the end after all the compute.
**Fix:** ASCII only: `status = "PASS" if pct >= GATE else "FAIL"`.

### 22. Template lambdas capture the loop variable
**Where:** if templates were built in a loop.
**Original code:**
```python
templates = []
for name, s in raw:
    templates.append((name, lambda p: s.format(p)))   # s is late-bound
```
**Error / problem:** every lambda ends up using the *last* `s` (Python closures capture by
reference), so all templates format the same way.
**Fix:** the repo defines them explicitly / binds via default arg:
```python
TEMPLATES = [("bare", lambda p: p + " "),
             ("discourse", lambda p: f"Context: {p}\nConclusion: ")]
```

### 23. Newline written as a literal in the template
**Where:** template definitions.
**Original code:**
```python
("discourse", lambda p: f"Context: {p}\\nConclusion: ")   # double backslash
```
**Error / problem:** `\\n` writes the two characters backslash-n, not a real newline, so the
discourse framing the model sees is wrong.
**Fix:** single `\n`: `f"Context: {p}\nConclusion: "`.

### 24. Comparing scores with `>=` instead of strict order
**Where:** `analyze()` strict-order count.
**Original code:**
```python
if c["prag"] >= c["impl"] >= c["lit"]:
```
**Error / problem:** ties count as "correct ordering", inflating the pass rate above the
true strict `prag > impl > lit`.
**Fix:** strict `>`:
```python
if c["prag"] > c["impl"] > c["lit"]:
```

### 25. Walrus on a possibly-missing condition
**Where:** `analyze()`.
**Original code:**
```python
ordered = sum(1 for r in results if r["conditions"][tname]["prag"] > ...)
```
**Error / problem:** if a template is missing for some item → `KeyError`.
**Fix:** guard with `.get()` + walrus:
```python
sum(1 for r in results if (c := r["conditions"].get(tname)) and c["prag"] > c["impl"] > c["lit"])
```

### 26. JSON dump crashes on non-ASCII premises
**Where:** saving `prescreen_results.json`.
**Original code:**
```python
with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)
```
**Error / problem:** on Windows the file opens as GBK; a premise with a special character →
`UnicodeEncodeError` while writing.
**Fix:** force UTF-8 (and keep unicode readable):
```python
with open(RESULTS_OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### 27. `Counter` / `defaultdict` used but not imported
**Where:** dataset inspection in `load_triplets()`.
**Original code:**
```python
rel_counts = Counter(...)
groups = defaultdict(dict)
```
**Error / problem:** `NameError: name 'Counter' is not defined`.
**Fix:** `from collections import Counter, defaultdict`.

### 28. VRAM creeps up across 100 items
**Where:** `run_prescreen()` scoring loop.
**Original code:**
```python
for idx, trip in enumerate(triplets):
    ...  # no cleanup
```
**Error / problem:** cached tensors accumulate; long runs slowly OOM.
**Fix:** periodic cleanup:
```python
if (idx + 1) % 50 == 0:
    gc.collect(); torch.cuda.empty_cache()
```

### 29. `N_MAX` slice on a list that's already short
**Where:** `return triplets[:N_MAX]`.
**Original code:**
```python
N_MAX = 400
return triplets[:N_MAX]
```
**Error / problem:** not an error, but a beginner expects 400 items and only gets ~100
(there are only 100 premises) — then later code that hard-codes 400 misbehaves.
**Fix:** derive counts from `len(...)`, don't assume `N_MAX`.

### 30. Gate check passes on an empty result set
**Where:** `analyze()`.
**Original code:**
```python
pct = ordered / total
```
**Error / problem:** if `total == 0` (bad load) → `ZeroDivisionError`.
**Fix:** `pct = ordered / total if total else 0`.

---

## Stage 1b — Multi-model scaling (`stage1_multimodel/run_multimodel_stage1.py`)

### 31. Reusing Stage 1 data under the wrong key
**Where:** `extract_1b4_stats()`.
**Original code:**
```python
cond = item["templates"].get(tname)   # stage1 file uses "conditions"
```
**Error / problem:** the 1.4B file stores results under `"conditions"`, not `"templates"`
(that key is only in the new 1b files) → `KeyError`.
**Fix:** read `item["conditions"].get(tname)` for the 1.4B file.

### 32. 410M still in VRAM when 2.8B loads
**Where:** model loop in `main()`.
**Original code:**
```python
for tag, name in MODELS:
    model = load_model(name)
    result = run_model_prescreen(model, items, tag)
    # no unload
```
**Error / problem:** loading 2.8B on top of 410M → OOM.
**Fix:** unload between models:
```python
unload_model(model)   # del + gc.collect + empty_cache
```

### 33. `except` block references `model` that may not exist
**Where:** the try/except around each model run.
**Original code:**
```python
except Exception as e:
    unload_model(model)   # model may have failed to load
```
**Error / problem:** if `load_model` itself threw, `model` is undefined →
`UnboundLocalError` hides the real error.
**Fix:** guard it: `unload_model(model) if 'model' in dir() else None`.

### 34. `matplotlib.use("Agg")` set too late
**Where:** top of the file.
**Original code:**
```python
import matplotlib.pyplot as plt
matplotlib.use("Agg")
```
**Error / problem:** backend must be chosen before `pyplot` is imported; on a headless
machine you get `... no display name and no $DISPLAY environment variable`.
**Fix:** set the backend first:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

### 35. Population vs sample std confusion in error bars
**Where:** aggregate stats.
**Original code:**
```python
"std_prag_margin": round(float(margins.std(ddof=1)), 4)
```
**Error / problem:** mixing `ddof=1` here and `np.std` (ddof=0) elsewhere makes the error
bars inconsistent between scripts.
**Fix:** pick one; the repo uses `np.std(margins)` (population) everywhere.

### 36. Bar-chart offsets overlap
**Where:** `plot_comparison()`.
**Original code:**
```python
bars = ax.bar(x, rates, width, ...)   # both series at same x
```
**Error / problem:** discourse and bare bars draw on top of each other.
**Fix:** offset each series:
```python
offset = (i - 0.5) * width
ax.bar(x + offset, rates, width, ...)
```

### 37. `success` counted as truthy floats
**Where:** `run_model_prescreen()`.
**Original code:**
```python
"success": prag_margin   # a float, not a bool
successes = [it["templates"][t]["success"] for it in results]
sum(successes)           # sums margins, not counts
```
**Error / problem:** summing floats instead of booleans gives a meaningless "success rate".
**Fix:** store a bool: `"success": prag_margin > PRAG_MARGIN_THRESHOLD`.

### 38. Plotting with only one model available
**Where:** end of `main()`.
**Original code:**
```python
plot_comparison(ordered_stats, ...)   # ordered_stats may have 1 entry
```
**Error / problem:** if 2.8B was skipped, the x-axis / grouping math breaks.
**Fix:** guard: `if len(ordered_stats) >= 2: plot_comparison(...)`.

### 39. `torch.no_grad()` redundant but harmless — the real miss is dtype
**Where:** `score_conditional()` copy.
**Original code:**
```python
model = HookedTransformer.from_pretrained(name, device="cuda")   # fp32
```
**Error / problem:** 2.8B in fp32 needs ~11 GB → OOM; the comment says ~5.6 GB which only
holds in bf16.
**Fix:** always pass `dtype=torch.bfloat16`.

### 40. Hard-coding the model order for the summary table
**Where:** summary print in `main()`.
**Original code:**
```python
for s in all_results_stats:   # insertion order: 410m, 2b8, then 1b4 appended
```
**Error / problem:** 1.4B is added last, so the table prints out of scale order (410M,
2.8B, 1.4B) which looks wrong for a scaling story.
**Fix:** reorder explicitly: `order = ["410m", "1b4", "2b8"]`.

---

## Stage 2 — Linear probing (`stage2_probing/probe.py`)

### 41. AUROC comes out 1.000 everywhere
**Where:** activation extraction design.
**Original code:**
```python
full_ids = model.to_tokens(prefix + hypothesis)
last = full_ids.shape[1] - 1     # end of the hypothesis
```
**Error / problem:** the hypotheses start with different words ("Not"/"All"/"No"), so a probe
on the hypothesis-end activation just detects which words are present → AUROC ≈ 1.0. It's a
leak, not a result.
**Fix:** extract at the **last prefix token only**, before any hypothesis text:
```python
prefix_ids = model.to_tokens(prefix, prepend_bos=True)
last = prefix_ids.shape[1] - 1
```

### 42. StandardScaler fit on the whole dataset
**Where:** `probe_binary()`.
**Original code:**
```python
X = StandardScaler().fit_transform(X)   # fit before CV split
for tr, va in skf.split(X, y): ...
```
**Error / problem:** validation stats leak into scaling → inflated AUROC.
**Fix:** fit inside each fold on train only:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X[tr]); X_val = scaler.transform(X[va])
```

### 43. `roc_auc_score` on a single-class fold
**Where:** `probe_binary()`.
**Original code:**
```python
for tr, va in skf.split(X, y):
    ...
    aurocs.append(roc_auc_score(y[va], prob))
```
**Error / problem:** imbalanced labels → a val fold can be all one class; AUROC is undefined
(older sklearn raises `ValueError: Only one class present`; newer returns `nan`) and poisons
the mean.
**Fix:** skip those folds:
```python
if len(np.unique(y[va])) < 2:
    continue
```

### 44. bfloat16 activation → `.numpy()`
**Where:** `extract_prefix_resid()`.
**Original code:**
```python
cache[f"blocks.{l}.hook_resid_post"][0, last, :].numpy()
```
**Error / problem:** `TypeError: Got unsupported ScalarType BFloat16` — NumPy has no bf16.
**Fix:** upcast first: `.float().numpy()`.

### 45. Caching all hooks on GPU during extraction
**Where:** `extract_prefix_resid()`.
**Original code:**
```python
_, cache = model.run_with_cache(prefix_ids)
```
**Error / problem:** stores every hook for every layer on the GPU, 100× → OOM.
**Fix:** filter + CPU:
```python
_, cache = model.run_with_cache(prefix_ids,
    names_filter=lambda n: n.endswith("hook_resid_post"), device="cpu")
```

### 46. Stacking activations of different lengths
**Where:** collecting per-item activations.
**Original code:**
```python
acts[tname].append(cache_full_sequence)   # (seq, d_model), seq varies
X = np.stack(acts[tname])                  # ragged
```
**Error / problem:** prefixes have different token counts, so full-sequence arrays can't
stack → `ValueError: all input arrays must have the same shape`.
**Fix:** take a single fixed position (last prefix token) per layer, giving a constant
`(24, 2048)` per item.

### 47. LogisticRegression doesn't converge
**Where:** `probe_binary()`.
**Original code:**
```python
clf = LogisticRegression()
```
**Error / problem:** default `max_iter=100` on 2048-dim inputs →
`ConvergenceWarning: lbfgs failed to converge`, unstable AUROC.
**Fix:** `LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")`.

### 48. No shuffling / no fixed seed in CV
**Where:** `StratifiedKFold`.
**Original code:**
```python
skf = StratifiedKFold(n_splits=5)
```
**Error / problem:** without `shuffle`, folds follow dataset order (items are grouped), and
without a seed results aren't reproducible run to run.
**Fix:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### 49. `predict_proba[:, 1]` when a class is missing in train
**Where:** `probe_binary()`.
**Original code:**
```python
prob = clf.predict_proba(X_val)[:, 1]
```
**Error / problem:** if a train fold saw only one class, `predict_proba` has a single column
→ `IndexError: index 1 is out of bounds`.
**Fix:** the single-class-fold skip (entry 43) prevents this; otherwise select by
`clf.classes_`.

### 50. Label built from text instead of behavior
**Where:** label construction.
**Original code:**
```python
y = np.array([1 if "Not" in item["prag_hyp"] else 0 for item in items])
```
**Error / problem:** that labels the *text*, so again the probe just reads token identity.
**Fix:** derive labels from Stage-1 scores:
```python
strict = np.array([1 if s["prag"] > s["impl"] > s["lit"] else 0 for s in sc])
```

### 51. Looping layers but indexing the wrong axis
**Where:** per-layer probing.
**Original code:**
```python
Xi = X[layer]          # X is (n_items, 24, 2048)
```
**Error / problem:** `X[layer]` picks an *item*, not a layer → wrong shape / wrong probe.
**Fix:** `Xi = X[:, layer, :]`.

### 52. Saving int-keyed dict to JSON
**Where:** saving `probe_results.json`.
**Original code:**
```python
probe_results[tname][label][layer] = auroc   # layer is an int key
json.dump(probe_results, f)
```
**Error / problem:** JSON keys must be strings; ints get silently coerced, and reloading
gives string keys you don't expect — subtle downstream mismatch.
**Fix:** be explicit (`str(layer)`) and remember keys are strings on reload.

---

## Stage 3 — Activation patching (`stage3_patching/patch_resid.py`, `patch_heads.py`)

### 53. Patch value stays on CPU
**Where:** `patch_resid.py`, building the hook.
**Original code:**
```python
pv = src_cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]   # on CPU
def _fn(val, hook):
    val[:, pos, :] = pv
```
**Error / problem:** `val` is on CUDA but `pv` came from a CPU cache →
`RuntimeError: Expected all tensors to be on the same device`.
**Fix:** move it: `pv = ...[0, -1, :].to("cuda")`.

### 54. Hook closure captures the loop variable late
**Where:** building per-layer / per-head hooks in a loop.
**Original code:**
```python
for layer in range(N_LAYERS):
    def _fn(val, hook):
        val[:, tgt_pos, :] = pv      # pv/layer bound by reference
```
**Error / problem:** all hooks end up using the *last* layer's `pv` (late binding).
**Fix:** bind with default args:
```python
def make_hook(pv=pv, pos=tgt_pos):
    def _fn(val, hook):
        val[:, pos, :] = pv
        return val
    return _fn
```

### 55. Hook doesn't return the tensor
**Where:** any patch hook.
**Original code:**
```python
def _fn(val, hook):
    val[:, pos, :] = pv
    # no return
```
**Error / problem:** TransformerLens expects the hook to return the (modified) activation;
returning `None` can drop the edit / error depending on version.
**Fix:** `return val` at the end.

### 56. Patching the wrong sequence position
**Where:** choosing `tgt_pos`.
**Original code:**
```python
val[:, 0, :] = pv        # patched BOS position
```
**Error / problem:** the causal signal we move is at the *last prefix token*, not BOS →
effect ≈ 0, wrong conclusion.
**Fix:** `tgt_pos = tgt_ids.shape[1] - 1` and patch `val[:, tgt_pos, :]`.

### 57. Copying the source position wrong
**Where:** reading the clean value.
**Original code:**
```python
pv = src_cache[...][0, tgt_pos, :]    # index src with the target's length
```
**Error / problem:** source (discourse) and target (bare) have different lengths, so
`tgt_pos` may be out of range or the wrong token in the source.
**Fix:** take the source's own last token: `src_cache[...][0, -1, :]`.

### 58. Dividing the normalized effect by ~0
**Where:** effect computation.
**Original code:**
```python
effect = (m_pat - m_tgt) / (m_src - m_tgt)
```
**Error / problem:** when discourse ≈ bare, the denominator is tiny → effects like `+37`,
which dominate `np.mean` and destroy the per-layer signal.
**Fix:** filter small denominators:
```python
effect = (m_pat - m_tgt)/denom if abs(denom) >= MIN_DENOM else None
```
then average only non-None values.

### 59. Averaging with `None`s still in the list
**Where:** summarizing per layer.
**Original code:**
```python
mean = float(np.mean(layer_data[layer]))
```
**Error / problem:** the filtered-out items are `None`; `np.mean` on a list with `None` →
`TypeError`.
**Fix:** drop them first: `valid = [e for e in layer_data[layer] if e is not None]`.

### 60. `prag_margin` uses the wrong baseline
**Where:** `prag_margin()`.
**Original code:**
```python
return s["prag"] - s["impl"]      # ignores lit
```
**Error / problem:** the margin should beat *both* distractors; using only `impl` overstates
success on items where `lit` is actually highest.
**Fix:** `return s["prag"] - max(s["impl"], s["lit"])`.

### 61. Recomputing clean/corrupted baselines inside the layer loop
**Where:** `patch_resid.py` main loop.
**Original code:**
```python
for layer in range(N_LAYERS):
    b_src = {k: score(model, src_ids, hyp_ids[k]) for k in hyps}   # every layer
```
**Error / problem:** the source/target baselines don't depend on the layer, so this reruns
48 extra forward passes per item — hugely slow.
**Fix:** compute `b_src`, `b_tgt`, `denom` once per item, before the layer loop.

### 62. Not freeing the source cache each item
**Where:** end of the item loop.
**Original code:**
```python
# next item, src_cache from before still alive
```
**Error / problem:** the resid cache for every item stacks up → OOM after a few dozen items.
**Fix:** `del src_cache; gc.collect(); torch.cuda.empty_cache()`.

### 63. Using `hook_result` for per-head patching (missing in fork)
**Where:** `patch_heads.py`.
**Original code:**
```python
target = {f"blocks.{l}.attn.hook_result" for l in TARGET_LAYERS}
```
**Error / problem:** the RelP fork doesn't compute `hook_result` by default →
`KeyError: 'blocks.1.attn.hook_result'` (and enabling it is memory-heavy).
**Fix:** patch `hook_z` (pre-`W_O`), shape `(batch, seq, n_heads, d_head)`:
```python
target_hook_names = {f"blocks.{l}.attn.hook_z" for l in TARGET_LAYERS}
```

### 64. Wrong axis order on `hook_z`
**Where:** `patch_heads.py` indexing.
**Original code:**
```python
src_z = src_cache[hz][0, -1, :]          # (n_heads*d_head?) unclear
val[:, pos, :] = src_z[head]
```
**Error / problem:** `hook_z` is `(batch, seq, n_heads, d_head)`; if you flatten or index the
wrong axis you patch garbage or hit a shape error.
**Fix:**
```python
src_z = src_cache[hz][0, -1, :, :]        # (n_heads, d_head)
val[:, pos, head, :] = src_z[head, :]
```

### 65. Assuming `d_head = 64`
**Where:** any manual reshape of head dims.
**Original code:**
```python
z = z.reshape(seq, 16, 64)   # wrong for pythia-1.4b
```
**Error / problem:** Pythia-1.4B has `d_model=2048`, 16 heads → `d_head=128`, not 64 →
`RuntimeError: shape ... is invalid for input of size ...`.
**Fix:** use `d_head = 128` (or read `model.cfg.d_head`).

### 66. Scanning all 24 layers for heads (slow / needless)
**Where:** `patch_heads.py` target layers.
**Original code:**
```python
TARGET_LAYERS = list(range(24))
```
**Error / problem:** 24×16 = 384 head-patches per item is very slow; Stage 3a already showed
the causal region is Layers 1–5.
**Fix:** `TARGET_LAYERS = list(range(1, 6))`.

### 67. `names_filter` lambda binds a changing set
**Where:** `run_with_cache` in `patch_heads.py`.
**Original code:**
```python
names_filter=lambda n: n in target_hook_names   # set reassigned later
```
**Error / problem:** if `target_hook_names` is rebound in the loop, the lambda sees the new
value → caches the wrong hooks.
**Fix:** bind it as a default: `lambda n, s=target_hook_names: n in s`.

### 68. Sorting "top heads" ascending
**Where:** `patch_heads.py` ranking.
**Original code:**
```python
top_heads.sort()
for mean, l, h in top_heads[:10]:
```
**Error / problem:** ascending sort lists the *weakest* heads first, so the "top 10" is
actually the bottom 10.
**Fix:** `top_heads.sort(reverse=True)`.

---

## Stage 4 — L01H03 deep analysis (`stage4_head_analysis/analyze_L01H03.py`)

### 69. Can't find the "some" token position
**Where:** `find_some_position()`.
**Original code:**
```python
tok_id = model.to_single_token("some")
```
**Error / problem:** in context "some" tokenizes with a leading space (`" some"`); the
no-space form may not match, so the search fails and attention-to-"some" is never measured.
**Fix:** try several surface forms:
```python
for candidate in [" some", "some", "Some", " Some"]:
    try: tok_id = model.to_single_token(candidate); break
    except Exception: continue
```

### 70. `to_single_token` throws on multi-token words
**Where:** anywhere a probe word is converted.
**Original code:**
```python
tid = model.to_single_token(word)   # word may be >1 token
```
**Error / problem:** `to_single_token` raises if the string isn't exactly one token →
uncaught `AssertionError/ValueError`.
**Fix:** wrap in try/except and fall back / skip that word.

### 71. Wrong hook name for attention pattern
**Where:** Experiment A.
**Original code:**
```python
names_filter=lambda n: n == f"blocks.{L}.attn.hook_attn"
```
**Error / problem:** the pattern hook is `hook_pattern`, not `hook_attn`; the cache key is
missing → `KeyError`.
**Fix:** `f"blocks.{TARGET_LAYER}.attn.hook_pattern"`.

### 72. Misreading the attention tensor shape
**Where:** Experiment A.
**Original code:**
```python
pattern = cache[hp][0, last_pos, TARGET_HEAD]   # wrong axis order
```
**Error / problem:** `hook_pattern` is `(batch, n_heads, q_pos, k_pos)`; head must come
before query. This indexes the wrong thing.
**Fix:** `cache[hp][0, TARGET_HEAD]` then take row `[last_pos]` for the query.

### 73. `np.mean` over "some" attention returns `nan`
**Where:** Experiment A aggregation.
**Original code:**
```python
mean_some = float(np.mean(some_attns))
```
**Error / problem:** items where "some" wasn't found are stored as `nan`; plain `mean` →
`nan` for the whole thing.
**Fix:** `np.nanmean(some_attns)`.

### 74. "other" bucket double-counts BOS/self/some
**Where:** Experiment A masking.
**Original code:**
```python
other = attn.sum() - attn[0]      # only removes BOS
```
**Error / problem:** self and "some" are still inside "other", so the buckets sum to more
than 1.0 and the interpretation is off.
**Fix:** build an explicit mask that removes BOS, self, and the some-position before summing.

### 75. DLA weights left in bfloat16
**Where:** Experiment B.
**Original code:**
```python
W_O = model.W_O[L, H]           # bf16, on GPU
resid = z @ W_O
```
**Error / problem:** bf16 matmuls lose precision for a vocab-wide projection, and mixing
CPU `z` with GPU `W_O` → device error.
**Fix:** pull to fp32 CPU once:
```python
W_O = model.W_O[L, H].float().cpu(); W_U = model.W_U.float().cpu()
```

### 76. Top-k tokens sliced in the wrong order
**Where:** Experiment B.
**Original code:**
```python
top_pos_idx = np.argsort(mean_dla)[:20]      # these are the smallest
```
**Error / problem:** `argsort` is ascending, so `[:20]` gives the most *suppressed* tokens
while labeling them "promoted".
**Fix:** `np.argsort(mean_dla)[-20:][::-1]` for the promoted ones.

### 77. Decoding token ids the wrong way
**Where:** turning ids into strings.
**Original code:**
```python
tok = model.to_string(i)          # i is a python int
```
**Error / problem:** `to_string` expects a tensor/list; a bare int can error or mis-decode.
**Fix:** `model.tokenizer.decode([i]).strip()`.

### 78. OV circuit matrices multiplied in the wrong order
**Where:** Experiment C.
**Original code:**
```python
W_OV = W_O @ W_V      # (d_head,d_model)@(d_model,d_head) -> wrong
```
**Error / problem:** shapes don't line up for the vocab projection / you get the transpose of
what you want → nonsense OV effects.
**Fix:** `W_OV = W_V @ W_O` giving `(d_model, d_model)`, then `W_E[tok] @ W_OV @ W_U`.

### 79. Mean-ablation hook uses a value on the wrong device
**Where:** Experiment D.
**Original code:**
```python
mean_z = torch.tensor(np.mean(all_z, axis=0))     # CPU float
val[:, pos, head, :] = mean_z                     # val on CUDA
```
**Error / problem:** device mismatch → `RuntimeError`.
**Fix:** `mean_z = torch.tensor(np.mean(all_z, 0), dtype=torch.float32).to("cuda")`.

### 80. Averaging `hook_z` across items of different lengths
**Where:** Experiment D step 1.
**Original code:**
```python
all_z.append(cache[hz][0, :, HEAD])    # whole sequence, ragged
np.mean(all_z, axis=0)
```
**Error / problem:** sequences differ in length → can't average.
**Fix:** take the last prefix token only: `cache[hz][0, last_pos, HEAD]` → each `(d_head,)`.

### 81. Zero-ablation compared to the wrong baseline
**Where:** Experiment D.
**Original code:**
```python
delta_zero = m_zero - m_mean     # compared against mean-ablation
```
**Error / problem:** both ablations should be measured against the *no-intervention*
baseline; comparing zero to mean gives a meaningless delta.
**Fix:** `delta_zero = np.mean(zero_margins - base_margins)`.

### 82. Ablation run on the discourse template
**Where:** Experiment D loop.
**Original code:**
```python
tgt_ids = model.to_tokens(TEMPLATE_SRC[1](premise), ...)   # discourse
```
**Error / problem:** the ablation question is "does L01H03 help in *bare*?" — running it on
discourse answers the wrong question.
**Fix:** use `TEMPLATE_TGT` (bare) for the ablation items.

### 83. Saving results before all experiments finish
**Where:** `main()`.
**Original code:**
```python
results["A"] = run_experiment_A(...)
json.dump(results, f)        # dumped early
results["B"] = run_experiment_B(...)
```
**Error / problem:** the saved JSON is missing B/C/D if the dump is misplaced.
**Fix:** collect all four, then dump once at the end.

### 84. Figure files overwrite each other
**Where:** each experiment's `savefig`.
**Original code:**
```python
fig.savefig(SAVE_DIR / "fig.png")     # same name every time
```
**Error / problem:** A, B, C, D all write `fig.png`; only the last survives.
**Fix:** unique names: `figA_attention_pattern.png`, `figB_DLA.png`, etc.

### 85. `plt.show()` blocks a headless run
**Where:** end of each experiment.
**Original code:**
```python
plt.show()
```
**Error / problem:** on a headless/Agg backend `show()` does nothing useful and can hang
scripted runs; figures also aren't closed → memory grows.
**Fix:** `fig.savefig(...); plt.close(fig)`.

### 86. Pragmatic-token DLA lookup misses space forms
**Where:** Experiment B `pragma_check`.
**Original code:**
```python
tid = model.to_single_token(word)      # "all" vs " all"
```
**Error / problem:** many words are one token only with a leading space, so the lookup fails
and the token is skipped.
**Fix:** try `[" " + word, word]` in order.

---

## Stage 4e — Attention breakdown (`stage4_head_analysis/analyze_4e_attn_breakdown.py`)

### 87. Newline token treated as `"\n"`
**Where:** `classify_positions()`.
**Original code:**
```python
newline_id = model.to_single_token("\n")
```
**Error / problem:** GPT-NeoX/Pythia is byte-level: newline is `Ċ`, space is `Ġ`. Looking up
`"\n"` can fail, so the `\nConclusion:` suffix isn't marked as `struct` and leaks into
`content` — which would flip the paper's structural-head conclusion.
**Fix:** try both forms:
```python
for candidate in ["\n", "Ċ"]:
    try: newline_id = model.to_single_token(candidate); break
    except Exception: continue
```

### 88. BOS and self counted twice
**Where:** `classify_positions()`.
**Original code:**
```python
cats["bos"].append(0)
cats["content"].append(0)   # 0 also swept into content later
```
**Error / problem:** without an `assigned` set, position 0 (and the last token) land in
multiple buckets, so category weights sum > 1.
**Fix:** track `assigned` and only put leftover positions in `content`.

### 89. Structural prefix positions matched by string, not ids
**Where:** `classify_positions()` struct prefix.
**Original code:**
```python
if model.to_string(prefix_ids[0, i]) == "Context":
```
**Error / problem:** decoding per-position and string-comparing is fragile (spaces, casing);
positions get missed.
**Fix:** tokenize `"Context: "` once and match ids by position:
```python
struct_prefix_ids = model.to_tokens("Context: ", prepend_bos=False)[0].tolist()
if ids_list[i] == struct_prefix_ids[i-1]: cats["struct"].append(i)
```

### 90. Taking every "some" occurrence
**Where:** quantifier classification.
**Original code:**
```python
for pos in range(seq_len):
    if ids_list[pos] == some_id:
        cats["quant"].append(pos)
```
**Error / problem:** if "some" appears twice, both are tagged and the bucket double-counts.
**Fix:** take the first only, then `break`.

### 91. Ratio divides by zero
**Where:** struct/content ratio.
**Original code:**
```python
ratio = mean_disc[struct] / mean_disc[content]
```
**Error / problem:** if content attention is ~0 → `ZeroDivisionError` / `inf`.
**Fix:** add epsilon: `mean_disc[struct] / (mean_disc[content] + 1e-9)`.

### 92. Reading `hook_pattern` with head after query
**Where:** `extract_attention_by_category()`.
**Original code:**
```python
attn = cache[hook][0, last_pos, TARGET_HEAD]
```
**Error / problem:** axis order is `(batch, n_heads, q, k)`; this indexes query before head →
wrong slice.
**Fix:** `cache[hook][0, TARGET_HEAD, last_pos]`.

### 93. Interpretation flag compares the wrong buckets
**Where:** final `interpretation`.
**Original code:**
```python
"interpretation": "structural_head" if mean_disc[bos] > mean_disc[content] else ...
```
**Error / problem:** the claim is about `struct` vs `content`, not `bos` — wrong label.
**Fix:** compare `mean_disc[struct_idx] > mean_disc[content_idx]`.

### 94. Bare template expected to have struct tokens
**Where:** bucket setup for bare.
**Original code:**
```python
# assumes struct positions exist in bare too
```
**Error / problem:** bare has no "Context:/Conclusion:", so `struct` must come out 0; code
that assumes otherwise mislabels ordinary premise words as struct.
**Fix:** only run the struct-matching branch when `template_type == "discourse"`.

---

## Figures, saving & workflow

### 95. Figure script can't find the result JSONs
**Where:** `figures/make_all_figures.py`.
**Original code:**
```python
with open("stage2_probing/probe_results.json") as f:   # relative to cwd
```
**Error / problem:** run from another folder → `FileNotFoundError`.
**Fix:** anchor paths to the file:
```python
ROOT = Path(__file__).parent.parent
open(ROOT / "stage2_probing" / "probe_results.json")
```

### 96. Reloaded JSON layer keys are strings
**Where:** plotting probe curves.
**Original code:**
```python
data = json.load(f)
y = [data[tname][label][layer] for layer in range(24)]   # int keys
```
**Error / problem:** JSON turned the layer keys into strings, so `data[...][layer]` (int) →
`KeyError`.
**Fix:** index with `str(layer)`.

### 97. Chinese labels show as boxes in matplotlib
**Where:** any figure with CJK text.
**Original code:**
```python
ax.set_title("话语框架优势")
```
**Error / problem:** default matplotlib fonts have no CJK glyphs → tofu boxes / warnings.
**Fix:** set a CJK-capable font (`plt.rcParams["font.sans-serif"] = ["SimHei"]`) or keep
figure text in English.

### 98. Windows path written with unescaped backslashes
**Where:** any hard-coded path.
**Original code:**
```python
path = "C:\Users\zenop\Desktop\MLLM\out.json"
```
**Error / problem:** `\U`, `\z` etc. are read as escape sequences →
`SyntaxError: (unicode error) ... \UXXXXXXXX`.
**Fix:** raw string or `pathlib`: `Path(r"C:\Users\zenop\Desktop\MLLM") / "out.json"`.

### 99. Committing the venv and model files to git
**Where:** first `git add .`.
**Original code:**
```bash
git add .
git commit -m "init"
```
**Error / problem:** `.venv-mi/` and cached model weights are huge → the push fails / repo
bloats past GitHub limits.
**Fix:** add a `.gitignore` for `.venv-mi/`, `*.pt`, HF cache, then commit.

### 100. Accidental AI signature in the commit message
**Where:** committing each stage.
**Original code:**
```bash
git commit -m "Add stage2" -m "Co-Authored-By: Claude <...>"
```
**Error / problem:** project rule forbids any `Co-Authored-By` / AI attribution in history.
**Fix:** plain message only: `git commit -m "Add stage2: linear probing"`.

### 101. `np.mean` over an empty valid list
**Where:** any per-layer/head summary after filtering.
**Original code:**
```python
mean = float(np.mean(valid))
```
**Error / problem:** if every item got filtered (all denoms tiny), `valid` is empty →
`RuntimeWarning: Mean of empty slice` and `nan`.
**Fix:** `mean = float(np.mean(valid)) if valid else float("nan")`.

### 102. Reusing one model object across stages without reloading
**Where:** running stages back-to-back in a notebook.
**Original code:**
```python
# stage2 leaves `model` around; stage3 reuses it after edits/hooks
```
**Error / problem:** leftover permanent hooks or a modified model from a previous stage
silently change later results.
**Fix:** `model.reset_hooks()` (or reload the model) at the start of each stage.
