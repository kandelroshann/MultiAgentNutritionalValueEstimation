from main import run_pipeline
import json
import glob

IMAGE_PATHS = [
    "/Users/roshankandel/NutritionAgentAdvisor/test/dish_1558029686.png",
    "/Users/roshankandel/NutritionAgentAdvisor/test/dish_1558031394.png",
    "/Users/roshankandel/NutritionAgentAdvisor/test/dish_1558114523.png",
    "/Users/roshankandel/NutritionAgentAdvisor/test/dish_1558382263.png"
]

for img in IMAGE_PATHS:
    print("\n===============================")
    print(f"Running pipeline on: {img}")
    print("===============================\n")

    out = run_pipeline(img)  # or run_pipeline(image_bytes)

    # Pretty-print JSON
    print(json.dumps(out, indent=4))

    # Save to file for appendix
    out_file = img.replace("images/", "").replace(".jpg", "_output.json")
    with open(f"outputs/{out_file}", "w") as f:
        json.dump(out, f, indent=4)