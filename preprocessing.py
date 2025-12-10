import os
import shutil

# Path to your main folder
main_folder = "./food-101/images"
output_folder = "./nutritionagentadvisor/images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each subfolder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Get all files in this subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        if files:
            # Sort files (optional, to make it deterministic)
            files.sort()
            first_image = files[0]
            src_path = os.path.join(subfolder_path, first_image)
            dst_path = os.path.join(output_folder, f"{subfolder}_{first_image}")
            shutil.copy(src_path, dst_path)
            print(f"Copied {first_image} from {subfolder}")
