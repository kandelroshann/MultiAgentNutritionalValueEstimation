import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from main import (
    process_single_image,
    process_folder_and_export,
    evaluate_folder_against_excel,
    create_chat_session_from_result,
)


def main() -> None:
    # Load env and ensure OpenAI key is present
    try:
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or env vars.")

    # Single image test
    single_image = "/Users/roshankandel/NutritionAgentAdvisor/134.jpg"
    single_result = process_single_image(single_image)
    print("Single image nutrition totals:")
    print(json.dumps(single_result.get("nutrition", {}).get("totals", {}), indent=2))
    
    # Chat: ask a follow-up about the same image
    chat = create_chat_session_from_result(single_result)
    followup_q = "what are the ingredients in this food?"
    answer = chat.ask(followup_q)
    print("Follow-up answer:")
    print(answer)

    # # Evaluate against ground-truth Excel
    #folder = "/Users/roshankandel/NutritionAgentAdvisor/images"
    #gt_excel = "/Users/roshankandel/NutritionAgentAdvisor/dishes.xlsx"
    #eval_result = evaluate_folder_against_excel(folder, gt_excel, tolerance=0.1)
    #print("Evaluation results:",eval_result)
    # # Per-image accuracies
    #for item in eval_result.get("per_image", []):
    #    print(f"{item.get('image_name','')}: {item.get('accuracy', 0.0):.3f}")
    # Overall accuracy
    #print(f"Overall numeric accuracy: {eval_result.get('overall_accuracy', 0.0):.3f}")

if __name__ == "__main__":
    main()