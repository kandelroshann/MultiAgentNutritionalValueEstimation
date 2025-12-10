#!/usr/bin/env python
"""
evaluation.py

Complete evaluation pipeline for your multi-agent nutrition system using
the Nutrition5k dataset, adapted for your exact file formats:

- dish_images.pkl
- dishes.xlsx
- dish_ingredients.xlsx
- ingredients.xlsx

This script performs:
1. Extraction of JPG images from dish_images.pkl
2. Building ground_truth.xlsx from dishes.xlsx
3. Running your nutrition pipeline on all images
4. Comparing predictions vs ground-truth
5. Computing accuracy metrics
6. Exporting eval_results.xlsx

Run:
    python evaluation.py -n /path/to/nutrition5k --out-excel results.xlsx
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from main import (
    evaluate_folder_against_excel,
    evaluate_folder_and_export,
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def ensure_api_key():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or export it.\n"
            "Example:\nexport OPENAI_API_KEY='sk-...'"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Nutrition Multi-Agent System")
    p.add_argument(
        "-n", "--nutrition5k-dir",
        required=True,
        help="Directory containing dish_images.pkl, dishes.xlsx, dish_ingredients.xlsx, ingredients.xlsx",
    )
    p.add_argument(
        "-o", "--out-excel",
        default="",
        help="Optional: save detailed evaluation to this Excel file",
    )
    p.add_argument(
        "--print-per-image",
        action="store_true",
        help="Print per-image accuracy",
    )
    return p.parse_args()


# ---------------------------------------------------------
# 1. Extract JPGs from dish_images.pkl
# ---------------------------------------------------------

def extract_images(nutrition_dir: Path) -> Path:
    """
    Extract dish images from dish_images.pkl into images/ folder.
    """
    images_dir = nutrition_dir / "images"
    images_dir.mkdir(exist_ok=True)

    pkl_path = nutrition_dir / "dish_images.pkl"
    df = pd.read_pickle(pkl_path)

    if not {"dish", "rgb_image"}.issubset(df.columns):
        raise ValueError(
            f"dish_images.pkl must contain columns ['dish', 'rgb_image'], found: {df.columns}"
        )

    for _, row in df.iterrows():
        dish = str(row["dish"])
        img_bytes = row["rgb_image"]

        out_path = images_dir / f"{dish}.jpg"
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(img_bytes)

    print(f"[INFO] Extracted images → {images_dir}")
    return images_dir


# ---------------------------------------------------------
# 2. Build ground_truth.xlsx from dishes.xlsx
# ---------------------------------------------------------

def build_ground_truth(nutrition_dir: Path) -> Path:
    """
    Convert the dishes.xlsx file into ground_truth.xlsx formatted for your evaluation code.
    """
    dishes_path = nutrition_dir / "dishes.xlsx"
    if not dishes_path.exists():
        raise FileNotFoundError(f"Missing: {dishes_path}")

    df = pd.read_excel(dishes_path)

    required_cols = [
        "dish_id", "total_calories", "total_fat", "total_carb", "total_protein"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in dishes.xlsx: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    gt = pd.DataFrame()
    gt["image_name"] = df["dish_id"].astype(str)
    gt["calories_kcal"] = df["total_calories"]
    gt["protein_g"] = df["total_protein"]
    gt["fat_g"] = df["total_fat"]
    gt["carbs_g"] = df["total_carb"]

    # Dataset does not include these → fill with 0
    gt["fiber_g"] = 0.0
    gt["sugar_g"] = 0.0
    gt["sodium_mg"] = 0.0

    out_path = nutrition_dir / "ground_truth.xlsx"
    gt.to_excel(out_path, index=False)

    print(f"[INFO] Ground truth file created → {out_path}")
    return out_path


# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------

def print_summary(eval_result: dict, print_per_image: bool = False):
    per_image = eval_result.get("per_image", [])
    overall = eval_result.get("overall_accuracy", 0.0)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total images evaluated: {len(per_image)}")
    print(f"Overall accuracy:       {overall:.3f}")

    if not per_image:
        print("No images matched ground truth.")
        return

    accs = [x["accuracy"] for x in per_image]
    print(f"Min accuracy:           {min(accs):.3f}")
    print(f"Max accuracy:           {max(accs):.3f}")
    print(f"Mean accuracy:          {sum(accs)/len(accs):.3f}")
    print("========================================\n")

    if print_per_image:
        print("Per-image results:")
        for item in per_image:
            print(f"{item['image_name']}: accuracy = {item['accuracy']:.3f}")


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    args = parse_args()
    ensure_api_key()

    nutrition_dir = Path(args.nutrition5k_dir).resolve()

    print("\n====== Preparing Dataset ======")
    images_dir = extract_images(nutrition_dir)
    gt_path = build_ground_truth(nutrition_dir)

    print("\n====== Running Evaluation ======")

    if args.out_excel:
        result = evaluate_folder_and_export(
            images_folder=str(images_dir),
            gt_excel_path=str(gt_path),
            out_excel_path=args.out_excel
        )
        print(f"[INFO] Detailed results saved to: {args.out_excel}")
    else:
        result = evaluate_folder_against_excel(
            images_folder=str(images_dir),
            gt_excel_path=str(gt_path)
        )

    print_summary(result, print_per_image=args.print_per_image)


if __name__ == "__main__":
    main()