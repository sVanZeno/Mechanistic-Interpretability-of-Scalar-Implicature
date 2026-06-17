import torch
import gc
import transformer_lens
from transformer_lens import HookedTransformer


def check_packages():
    import transformers, datasets
    print(f"torch       {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"datasets    {datasets.__version__}")
    tl_ver = getattr(transformer_lens, "__version__", "RelP-fork")
    print(f"transformer_lens {tl_ver}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024**3
        print(f"GPU: {props.name}  VRAM: {total_vram:.1f} GB")


def smoke_test_160m():
    print("\n--- Pythia-160M smoke test ---")
    model = HookedTransformer.from_pretrained(
        "pythia-160m",
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    torch.set_grad_enabled(False)

    tokens = model.to_tokens("Hello world")
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda n: n.endswith("hook_resid_post"),
        device="cpu",
    )

    assert "blocks.0.hook_resid_post" in cache, "cache key missing"
    print(f"cache keys (first 3): {list(cache.keys())[:3]}")
    print(f"blocks.0.hook_resid_post shape: {cache['blocks.0.hook_resid_post'].shape}")
    print("Pythia-160M smoke test PASSED")

    del model, cache
    gc.collect()
    torch.cuda.empty_cache()


def vram_test_1_4b():
    if not torch.cuda.is_available():
        print("\nNo CUDA device — skipping Pythia-1.4B VRAM test")
        return

    print("\n--- Pythia-1.4B VRAM test ---")
    torch.cuda.reset_peak_memory_stats()

    model = HookedTransformer.from_pretrained(
        "pythia-1.4b",
        dtype=torch.bfloat16,
        device="cuda",
    )
    model.eval()
    torch.set_grad_enabled(False)

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Free:      {total - reserved:.2f} GB  (of {total:.1f} GB total)")

    if allocated < 5.0:
        print("VRAM check PASSED (bf16 weight footprint within expected range)")
    else:
        print("WARNING: allocated VRAM higher than expected — check for fp32 fallback")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    check_packages()
    smoke_test_160m()
    vram_test_1_4b()
    print("\nStage 0 complete.")
