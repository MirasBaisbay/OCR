"""
Run this on your Windows machine to inspect the exact PaddleOCR-VL output structure.
Usage: python debug_paddle.py <path_to_image_or_pdf>
"""
import os, sys, json

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCRVL

def deep_inspect(obj, prefix="", depth=0, max_depth=4):
    """Recursively inspect object structure."""
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix} ... (max depth)")
        return

    if obj is None:
        print(f"{indent}{prefix} None")
        return

    t = type(obj).__name__

    if isinstance(obj, (str, int, float, bool)):
        val = str(obj)[:150]
        print(f"{indent}{prefix} ({t}) = {val}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{prefix} ({t}) len={len(obj)}")
        for i, item in enumerate(obj[:3]):  # show first 3
            deep_inspect(item, prefix=f"[{i}]", depth=depth+1, max_depth=max_depth)
        if len(obj) > 3:
            print(f"{indent}  ... +{len(obj)-3} more items")
    elif isinstance(obj, dict):
        print(f"{indent}{prefix} (dict) keys={list(obj.keys())}")
        for k, v in obj.items():
            deep_inspect(v, prefix=f".{k}", depth=depth+1, max_depth=max_depth)
    else:
        # Object with attributes
        print(f"{indent}{prefix} ({t})")
        attrs = [a for a in dir(obj) if not a.startswith('_')]
        # Filter to likely data attributes
        for a in attrs:
            try:
                val = getattr(obj, a)
                if callable(val):
                    continue
                deep_inspect(val, prefix=f".{a}", depth=depth+1, max_depth=max_depth)
            except Exception:
                print(f"{indent}  .{a} = <error reading>")


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"

    print(f"Input: {img_path}\n")
    pipeline = PaddleOCRVL()
    output = list(pipeline.predict(img_path))

    for i, res in enumerate(output):
        print(f"\n{'='*70}")
        print(f"Result [{i}] — top-level keys: {list(res.keys())}")
        print(f"{'='*70}")

        for key in ["layout_det_res", "parsing_res_list", "table_res_list",
                     "spotting_res", "markdown", "html"]:
            val = res.get(key)
            print(f"\n--- {key} ---")
            deep_inspect(val, prefix=key, depth=0, max_depth=3)