#!/usr/bin/env python
"""
ingredient_evaluation.py

Research-grade evaluation of ingredient detection for your multi-agent
LLM nutrition system using the Nutrition5k dataset.

This version includes DISK CACHING to avoid repeated OpenAI calls:
  - First run: calls ingredients_agent() and saves predictions to JSON
  - Subsequent runs: loads predictions from cache instead of calling the API

Could not be implemented completely. 
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
import difflib
import json

# --- Import from your main project ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from main import (  # type: ignore[import-not-found]
    load_image,
    ingredients_agent,
    AppState,
)


# ---------------------------------------------------------
# ENV + ARGS
# ---------------------------------------------------------

def ensure_api_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or export it.\n"
            "Example:\nexport OPENAI_API_KEY='sk-...'"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Research-grade ingredient detection evaluation (with caching)")
    p.add_argument(
        "-n", "--nutrition5k-dir",
        required=True,
        help="Path to Nutrition5k folder (containing dish_ingredients.xlsx and images/).",
    )
    p.add_argument(
        "-o", "--out-csv",
        default="ingredient_eval_results.csv",
        help="Where to save the detailed per-dish evaluation CSV.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional: limit number of images for cost/time control.",
    )
    p.add_argument(
        "--print-samples",
        action="store_true",
        help="Print a few good and bad sample matches.",
    )
    return p.parse_args()


# ---------------------------------------------------------
# GROUND TRUTH INGREDIENTS
# ---------------------------------------------------------

def normalize_name(name: str) -> str:
    """Normalize ingredient names for comparison."""
    import re
    s = (name or "").strip().lower()
    # remove punctuation
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_ground_truth_ingredients(nutrition_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load ground-truth ingredients from dish_ingredients.xlsx.

    Returns:
        mapping: dish_id (string) -> list of dicts:
            {
              "name": original ingredient name,
              "name_norm": normalized name,
              "grams": float grams
            }
    """
    path = nutrition_dir / "dish_ingredients.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing dish_ingredients.xlsx at {path}")

    df = pd.read_excel(path)

    required = {"dish_id", "ingr_name", "grams"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"dish_ingredients.xlsx must contain columns {required}, "
            f"found: {list(df.columns)}"
        )

    mapping: Dict[str, List[Dict]] = {}
    for _, row in df.iterrows():
        dish_id = str(row["dish_id"]).strip()
        if not dish_id:
            continue
        name = str(row["ingr_name"])
        grams = float(row.get("grams", 0.0) or 0.0)
        entry = {
            "name": name,
            "name_norm": normalize_name(name),
            "grams": grams,
        }
        if dish_id not in mapping:
            mapping[dish_id] = []
        mapping[dish_id].append(entry)

    return mapping


# ---------------------------------------------------------
# PREDICTION HELPERS (WITH CACHING)
# ---------------------------------------------------------

def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def get_cache_path(cache_dir: Path, dish_id: str) -> Path:
    """
    Return the path for the cached ingredient predictions for a given dish.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # one JSON per dish_id
    return cache_dir / f"{dish_id}.json"


def run_ingredients_only(image_path: Path, dish_id: str, cache_dir: Path) -> List[Dict]:
    """
    Run only the ingredients agent to save tokens, WITH DISK CACHING.

    Behavior:
      - If a cache file exists for this dish_id, load it and return ingredients
      - Otherwise, call ingredients_agent(), cache the result, then return ingredients

    Returns:
        List of ingredient dicts with keys:
          name, preparation, estimated_grams
    """
    cache_path = get_cache_path(cache_dir, dish_id)

    # ---- CACHE HIT ----
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # We store the list directly in cache: [ {name, preparation, estimated_grams}, ... ]
            if isinstance(cached, list):
                return cached
        except Exception as e:
            print(f"  [WARN] Failed to load cache for {dish_id}: {e} (will re-call API)")

    # ---- CACHE MISS: CALL OPENAI VIA ingredients_agent ----
    state: AppState = {"image_path": str(image_path)}  # type: ignore[assignment]
    # load image bytes + mime
    state.update(load_image(state))
    # ingredient agent (this calls OpenAI once)
    state.update(ingredients_agent(state))
    ingredients_obj = state.get("ingredients", {}) or {}
    ing_list = ingredients_obj.get("ingredients", [])
    if not isinstance(ing_list, list):
        ing_list = []

    # Save to cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ing_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [WARN] Failed to write cache for {dish_id}: {e}")

    return ing_list


# ---------------------------------------------------------
# MATCHING LOGIC
# ---------------------------------------------------------

def fuzzy_match_one(
    target: str,
    candidates: List[str],
    used: List[bool],
    threshold: float = 0.8,
) -> int:
    """
    Fuzzy-match a single target string to the best candidate.
    Returns index of best match in candidates (if >= threshold and not used),
    or -1 if no good match.
    """
    best_idx = -1
    best_score = 0.0
    for i, cand in enumerate(candidates):
        if used[i]:
            continue
        score = difflib.SequenceMatcher(None, target, cand).ratio()
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score >= threshold:
        return best_idx
    return -1


def compare_ingredients_for_dish(
    gt_ings: List[Dict],
    pred_ings: List[Dict],
    fuzzy_threshold: float = 0.8,
) -> Dict:
    """
    Compare ground-truth vs predicted ingredients for a single dish.

    Computes:
      - TP, FP, FN (count-based)
      - grams_gt_total
      - grams_matched (sum of grams of matched GT ingredients)
      - per-dish precision, recall, F1 (count)
      - gram_recall (grams_matched / grams_gt_total)
    """
    gt_names_norm = [ing["name_norm"] for ing in gt_ings]
    gt_grams = [ing["grams"] for ing in gt_ings]
    pred_names_norm = [normalize_name(ing.get("name", "")) for ing in pred_ings]

    gt_used = [False] * len(gt_names_norm)
    pred_used = [False] * len(pred_names_norm)

    tp = 0
    fp = 0
    fn = 0
    grams_gt_total = float(sum(gt_grams))
    grams_matched = 0.0

    # First, match each ground-truth ingredient to predictions
    for gi, gt_name in enumerate(gt_names_norm):
        match_idx = fuzzy_match_one(gt_name, pred_names_norm, pred_used, threshold=fuzzy_threshold)
        if match_idx >= 0:
            tp += 1
            pred_used[match_idx] = True
            gt_used[gi] = True
            grams_matched += gt_grams[gi]
        else:
            fn += 1

    # Remaining unmatched predicted ingredients are false positives
    for used in pred_used:
        if not used:
            fp += 1

    # Compute per-dish precision / recall / F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    gram_recall = grams_matched / grams_gt_total if grams_gt_total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_gt": len(gt_ings),
        "num_pred": len(pred_ings),
        "grams_gt_total": grams_gt_total,
        "grams_matched": grams_matched,
        "gram_recall": gram_recall,
    }


# ---------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------

def main():
    args = parse_args()
    ensure_api_key()

    nutrition_dir = Path(args.nutrition5k_dir).resolve()
    images_dir = nutrition_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images folder at {images_dir}. "
            "Make sure you've already extracted images from dish_images.pkl."
        )

    # Create cache directory inside Nutrition5k folder
    cache_dir = nutrition_dir / "ingredient_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using ingredient cache directory: {cache_dir}")

    print("Loading ground-truth ingredients...")
    gt_map = load_ground_truth_ingredients(nutrition_dir)
    print(f"Loaded GT ingredients for {len(gt_map)} dishes.")

    all_images = list_images(images_dir)
    if args.max_images is not None:
        all_images = all_images[: args.max_images]

    print(f"Found {len(all_images)} images to evaluate.")

    per_dish_records = []

    micro_tp = micro_fp = micro_fn = 0
    total_grams_gt = 0.0
    total_grams_matched = 0.0

    for idx, img_path in enumerate(all_images, start=1):
        image_name = img_path.name
        dish_id = image_name.rsplit(".", 1)[0]  # strip extension

        if dish_id not in gt_map:
            # some dishes may not have annotations or naming mismatch
            continue

        print(f"[{idx}/{len(all_images)}] Evaluating {image_name} (dish_id={dish_id})")

        gt_ings = gt_map[dish_id]

        try:
            pred_ings = run_ingredients_only(img_path, dish_id, cache_dir)
        except Exception as e:
            print(f"  ERROR processing {image_name}: {e}")
            continue

        metrics = compare_ingredients_for_dish(gt_ings, pred_ings, fuzzy_threshold=0.8)

        micro_tp += metrics["tp"]
        micro_fp += metrics["fp"]
        micro_fn += metrics["fn"]
        total_grams_gt += metrics["grams_gt_total"]
        total_grams_matched += metrics["grams_matched"]

        record = {
            "image_name": image_name,
            "dish_id": dish_id,
            "num_gt": metrics["num_gt"],
            "num_pred": metrics["num_pred"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "grams_gt_total": metrics["grams_gt_total"],
            "grams_matched": metrics["grams_matched"],
            "gram_recall": metrics["gram_recall"],
        }
        per_dish_records.append(record)

    # Convert to DataFrame and save
    df = pd.DataFrame(per_dish_records)
    out_csv_path = Path(args.out_csv).resolve()
    df.to_csv(out_csv_path, index=False)
    print(f"\n[INFO] Detailed per-dish ingredient evaluation saved to: {out_csv_path}")

    # Micro-averaged metrics
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    gram_recall_global = total_grams_matched / total_grams_gt if total_grams_gt > 0 else 0.0

    print("\n========== INGREDIENT DETECTION SUMMARY ==========")
    print(f"Images evaluated:             {len(per_dish_records)}")
    print(f"Micro TP / FP / FN:          {micro_tp} / {micro_fp} / {micro_fn}")
    print(f"Micro Precision:             {micro_precision:.3f}")
    print(f"Micro Recall:                {micro_recall:.3f}")
    print(f"Micro F1:                    {micro_f1:.3f}")
    print(f"Global Gram-weighted Recall: {gram_recall_global:.3f}")
    print("========================================\n")

    if args.print_samples and not df.empty:
        # Print a couple of best and worst dishes by F1
        df_sorted = df.sort_values("f1")
        print("---- Lowest F1 dishes (hardest cases) ----")
        print(df_sorted.head(5)[["image_name", "dish_id", "precision", "recall", "f1", "gram_recall"]])
        print("\n---- Highest F1 dishes (easiest cases) ----")
        print(df_sorted.tail(5)[["image_name", "dish_id", "precision", "recall", "f1", "gram_recall"]])


if __name__ == "__main__":
    main()
