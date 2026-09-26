"""Minimal JSON persistence for experiment scripts.

Usage (at end of a script):
    from results_io import save_results
    save_results(__file__, config={...}, results={...})

Writes to `results/<script_stem>.json` next to the script. Handles numpy arrays and
tuple-valued dict keys automatically.
"""
import json
import os
import numpy as np


def _sanitize(obj):
    if isinstance(obj, dict):
        return {(f"d={k[0]},r={k[1]}" if isinstance(k, tuple) and len(k) == 2 else str(k)):
                _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_results(script_file, config, results):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(script_file)), "results")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(script_file))[0]
    out_path = os.path.join(out_dir, stem + ".json")
    payload = {"config": _sanitize(config), "results": _sanitize(results)}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[results_io] Saved {out_path}")
    return out_path
