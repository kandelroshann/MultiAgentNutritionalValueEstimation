import pandas as pd
import numpy as np

df = pd.read_excel("/Users/roshankandel/NutritionAgentAdvisor/nutrition5k/results.xlsx")

print("\n=== BASIC STATS ===")
print(df.describe())

# List of nutrient pairs
pairs = [
    ("ref_calories_kcal", "pred_calories_kcal"),
    ("ref_protein_g", "pred_protein_g"),
    ("ref_fat_g", "pred_fat_g"),
    ("ref_carbs_g", "pred_carbs_g"),
]

for ref, pred in pairs:
    ref_vals = df[ref].astype(float)
    pred_vals = df[pred].astype(float)

    mae = np.mean(np.abs(pred_vals - ref_vals))
    mape = np.mean(np.abs((pred_vals - ref_vals) / (ref_vals + 1e-9))) * 100
    rmse = np.sqrt(np.mean((pred_vals - ref_vals)**2))

    print(f"\n=== {ref.replace('ref_', '').upper()} ===")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f}")

print("\n=== ACCURACY COLUMN SUMMARY ===")
if "accuracy" in df.columns:
    print(df["accuracy"].describe())
else:
    print("No 'accuracy' column found.")