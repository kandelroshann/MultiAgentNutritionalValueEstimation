from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# --- OpenAI client ---
from openai import OpenAI


class DishSchema(BaseModel):
    dish_name: str = Field(description="Canonical dish name, e.g., 'chicken biryani'")
    category: str = Field(description="Course: breakfast/lunch/dinner/snack/dessert/beverage")
    cuisine: str = Field(description="Likely cuisine, e.g., Indian, Italian")
    confidence: float = Field(ge=0, le=1)


class IngredientItem(BaseModel):
    name: str
    preparation: str = Field(default="")
    estimated_grams: float = Field(default=0.0)


class IngredientsSchema(BaseModel):
    ingredients: List[IngredientItem]


class QuantitySchema(BaseModel):
    total_estimated_grams: float
    rationale: str


class AppState(TypedDict, total=False):
    image_path: str
    image_bytes: bytes
    mime_type: str
    dish: Dict[str, Any]
    ingredients: Dict[str, Any]
    quantity: Dict[str, Any]
    nutrition: Dict[str, Any]


class OpenAIClient:
    def __init__(self, model: str = "") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or env vars.")
        self.client = OpenAI(
            api_key=api_key,
        )
        # Default to OpenAI gpt-4o-mini unless overridden
        self.model = "gpt-4.1"
        # self.model = "gpt-5-nano"

    def generate_json(self, prompt: str, image_bytes: bytes, schema: Dict[str, Any], mime_type: str = "image/jpeg") -> Dict[str, Any]:
        # Encode image to base64 Data URL per OpenAI-compatible schema
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64}"
        system = (
            "You are a precise vision assistant. Always reply with strict JSON conforming to the provided schema."
        )
        # Use response_format with schema if supported
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                },
            },
            temperature=0,
        )
        text = response.choices[0].message.content
        return json.loads(text)


class NutritionFields(BaseModel):
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float


class NutritionPerItemModel(BaseModel):
    name: str
    grams: float
    nutrition: NutritionFields


class NutritionSchema(BaseModel):
    per_item: List[NutritionPerItemModel]
    totals: NutritionFields


def load_image(state: AppState) -> AppState:
    import mimetypes

    image_path = state["image_path"]
    with open(image_path, "rb") as f:
        img = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    return {"image_bytes": img, "mime_type": mime_type}

# --- Agents ---

def dish_agent(state: AppState) -> AppState:
    client = OpenAIClient()
    schema = DishSchema.model_json_schema()
    prompt = (
        "You are a food image expert. Identify the dish from the image."
        " Return JSON with: dish_name, category, cuisine, confidence."
        " Avoid hallucination; if unsure, lower confidence."
    )
    result = client.generate_json(prompt, state["image_bytes"], schema, mime_type=state.get("mime_type", "image/jpeg"))
    return {"dish": result}


def ingredients_agent(state: AppState) -> AppState:
    client = OpenAIClient()
    schema = IngredientsSchema.model_json_schema()

    # NEW: use dish context from previous agent if available
    dish_ctx = state.get("dish", {})
    dish_hint = ""
    if dish_ctx:
        dish_hint = (
            "\nDish context from previous agent (use as a soft hint, not hard truth): "
            f"{json.dumps(dish_ctx, ensure_ascii=False)}"
            "\nIf the visual evidence conflicts with the dish name, trust the image."
        )

    prompt = (
        "You are a professional food-composition analyst. Carefully examine the ENTIRE dish."
        " Identify ALL food components, even if partially visible, including:"
        " • primary protein items (chicken, beef, fish, tofu, eggs)"
        " • secondary items (sauces, dressings, seasonings)"
        " • vegetables (raw or cooked)"
        " • grains, pasta, rice, bread"
        " • cooked mixtures (stir fry, curry, casserole)"
        " • garnishes and toppings."
        " For EACH ingredient, return:"
        "  - name"
        "  - preparation (e.g., cooked, grilled, raw, chopped, mixed, sauced)"
        "  - estimated_grams (based on visual volume relative to plate size)."
        " Estimate grams realistically using known serving size heuristics."
        " Return as many ingredients as necessary to explain the meal."
        f"{dish_hint}"
    )

    result = client.generate_json(
        prompt,
        state["image_bytes"],
        schema,
        mime_type=state.get("mime_type", "image/jpeg"),
    )
    return {"ingredients": result}

def quantity_agent(state: AppState) -> AppState:
    client = OpenAIClient()
    schema = QuantitySchema.model_json_schema()

    # NEW: use dish + ingredients as context
    dish_ctx = state.get("dish", {})
    ing_ctx = state.get("ingredients", {})
    context = {
        "dish": dish_ctx,
        "ingredients": ing_ctx,
    }

    prompt = (
        "Estimate the TOTAL edible mass on the plate in grams."
        " Provide total_estimated_grams and a brief rationale."
        " Use plate/utensil scale if visible and typical serving sizes."
        " Also consider the recognized dish type and ingredient list when judging portions."
        f"\nContext (dish + ingredients) as JSON: {json.dumps(context, ensure_ascii=False)}"
    )

    result = client.generate_json(
        prompt,
        state["image_bytes"],
        schema,
        mime_type=state.get("mime_type", "image/jpeg"),
    )
    return {"quantity": result}


def nutrition_agent(state: AppState) -> AppState:
    client = OpenAIClient()
    schema = NutritionSchema.model_json_schema()
    context = {
        "dish": state.get("dish", {}),
        "ingredients": state.get("ingredients", {}),
        "quantity": state.get("quantity", {}),
    }
    prompt = (
         
    "You are a food nutrition scientist. Aggregate all ingredients into a realistic nutrition profile."
    "Use standard USDA food composition equivalents. Do NOT under-estimate totals."
    "Ensure the sum of ingredient grams is close to total_estimated_grams."
    "Scale nutrition proportionally if total_estimated_grams differs from visible ingredient masses."
    "Never output unrealistically low calories—full meals must fall within typical ranges (200–1200 kcal)."
    f"Context: {json.dumps(context)}"
    )
    result = client.generate_json(
        prompt,
        state.get("image_bytes", b""),
        schema,
        mime_type=state.get("mime_type", "image/jpeg"),
    )
    print("Nutrition agent result totals:", result)
    return {"nutrition": result}


# --- Build graph ---
def build_graph():
    graph = StateGraph(AppState)
    graph.add_node("load_image", load_image)
    graph.add_node("dish_agent", dish_agent)
    graph.add_node("ingredients_agent", ingredients_agent)
    graph.add_node("quantity_agent", quantity_agent)
    graph.add_node("nutrition_agent", nutrition_agent)

    graph.add_edge(START, "load_image")
    graph.add_edge("load_image", "dish_agent")
    graph.add_edge("dish_agent", "ingredients_agent")
    graph.add_edge("ingredients_agent", "quantity_agent")
    graph.add_edge("quantity_agent", "nutrition_agent")
    graph.add_edge("nutrition_agent", END)
    return graph.compile()


# --- Programmatic API ---
def run_pipeline(image_path: str) -> Dict[str, Any]:
    """Run the multi-agent pipeline on a single image and return the full state."""
    network = build_graph()
    final_state = network.invoke({"image_path": image_path})
    # Ensure the original image path is present in the returned state for bookkeeping
    final_state["image_path"] = image_path
    return final_state


def process_single_image(image_path: str) -> Dict[str, Any]:
    """Convenience wrapper for processing one image from code."""
    return run_pipeline(image_path)


def export_nutrition_to_excel(results: List[Dict[str, Any]], excel_path: str) -> None:
    """Export a list of pipeline results to an Excel file with nutrition totals."""
    rows: List[Dict[str, Any]] = []
    for res in results:
        image_name = os.path.basename(res.get("image_path", ""))
        dish_name = (res.get("dish", {}) or {}).get("dish_name", "")
        totals = (res.get("nutrition", {}) or {}).get("totals", {})
        rows.append({
            "image_name": image_name,
            "dish_name": dish_name,
            "calories_kcal": totals.get("calories_kcal", 0.0),
            "protein_g": totals.get("protein_g", 0.0),
            "fat_g": totals.get("fat_g", 0.0),
            "carbs_g": totals.get("carbs_g", 0.0),
            "fiber_g": totals.get("fiber_g", 0.0),
            "sugar_g": totals.get("sugar_g", 0.0),
            "sodium_mg": totals.get("sodium_mg", 0.0),
        })
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)


def process_bulk_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple images and return a list of results (one per image)."""
    results: List[Dict[str, Any]] = []
    for path in image_paths:
        try:
            res = run_pipeline(path)
            results.append(res)
        except Exception as e:
            # Include a minimal failure record for traceability
            results.append({
                "image_path": path,
                "error": str(e),
            })
    return results


def process_bulk_and_export(image_paths: List[str], excel_path: str) -> None:
    """Process multiple images and export nutrition totals to an Excel file."""
    results = process_bulk_images(image_paths)
    export_nutrition_to_excel(results, excel_path)


def process_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """Run the pipeline directly from in-memory image bytes (no file path needed)."""
    state: Dict[str, Any] = {"image_bytes": image_bytes, "mime_type": mime_type}
    state.update(dish_agent(state))
    state.update(ingredients_agent(state))
    state.update(quantity_agent(state))
    state.update(nutrition_agent(state))
    return state


def list_images_in_folder(folder_path: str) -> List[str]:
    """Return a list of absolute image file paths from a folder (jpg/png/jpeg/webp)."""
    p = Path(folder_path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder_path}")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [str(f.resolve()) for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]


def process_folder_and_export(folder_path: str, excel_path: str) -> None:
    """Process all supported images in a folder and export nutrition totals to Excel."""
    images = list_images_in_folder(folder_path)
    process_bulk_and_export(images, excel_path)


# --- Evaluation helpers ---
def _extract_totals_from_row(row: pd.Series) -> Dict[str, Any]:
    fields = [
        "calories_kcal",
        "protein_g",
        "fat_g",
        "carbs_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
    ]
    return {k: row.get(k) for k in fields}


def load_ground_truth_excel(gt_excel_path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_excel(gt_excel_path)
    if "image_name" not in df.columns:
        raise ValueError("Ground-truth Excel must contain an 'image_name' column")
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row.get("image_name", "")).strip()
        if not name:
            continue
        # ✅ Use name *without* extension as the key
        base = os.path.splitext(name)[0]
        mapping[base] = _extract_totals_from_row(row)
    return mapping

def compute_numeric_accuracy(reference: Dict[str, Any], prediction: Dict[str, Any], tolerance: float = 0.1) -> float:
    fields = [
        "calories_kcal",
        "protein_g",
        "fat_g",
        "carbs_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
    ]
    scores: List[float] = []
    for f in fields:
        ref_val = reference.get(f)
        pred_val = prediction.get(f)
        try:
            ref_val = float(ref_val)
            pred_val = float(pred_val)
            rel_err = abs(ref_val - pred_val) / abs(ref_val) if ref_val else 0.0
            score = 1.0 if rel_err <= tolerance else max(0.0, 1 - rel_err / tolerance)
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_folder_against_excel(images_folder: str, gt_excel_path: str, tolerance: float = 0.1) -> Dict[str, Any]:
    gt_map = load_ground_truth_excel(gt_excel_path)
    image_paths = list_images_in_folder(images_folder)

    per_image: List[Dict[str, Any]] = []
    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        base = os.path.splitext(image_name)[0]  # ✅ strip .jpg/.png/etc
        if base not in gt_map:
            continue
        reference = gt_map[base]
        try:
            result = process_single_image(img_path)
            prediction = (result.get("nutrition", {}) or {}).get("totals", {})
            dish_name = (result.get("dish", {}) or {}).get("dish_name", "")
            acc = compute_numeric_accuracy(reference, prediction, tolerance=tolerance)
            per_image.append({
                "image_name": image_name,
                "dish_name": dish_name,
                "reference": reference,
                "prediction": prediction,
                "accuracy": acc,
            })
        except Exception as e:
            per_image.append({
            "image_name": image_name,
            "dish_name": "",
            "reference": reference,
            "prediction": {},
            "accuracy": 0.0,
            "error": str(e),
        })

    overall = sum(item.get("accuracy", 0.0) for item in per_image) / len(per_image) if per_image else 0.0
    return {"overall_accuracy": overall, "per_image": per_image}


def evaluate_folder_and_export(images_folder: str, gt_excel_path: str, out_excel_path: str, tolerance: float = 0.1) -> Dict[str, Any]:
    eval_result = evaluate_folder_against_excel(images_folder, gt_excel_path, tolerance=tolerance)
    # Flatten rows for export
    rows: List[Dict[str, Any]] = []
    for item in eval_result.get("per_image", []):
        pred = item.get("prediction", {}) or {}
        ref = item.get("reference", {}) or {}
        rows.append({
            "image_name": item.get("image_name", ""),
            "dish_name": item.get("dish_name", ""),
            # reference values
            "ref_calories_kcal": ref.get("calories_kcal", 0.0),
            "ref_protein_g": ref.get("protein_g", 0.0),
            "ref_fat_g": ref.get("fat_g", 0.0),
            "ref_carbs_g": ref.get("carbs_g", 0.0),
            "ref_fiber_g": ref.get("fiber_g", 0.0),
            "ref_sugar_g": ref.get("sugar_g", 0.0),
            "ref_sodium_mg": ref.get("sodium_mg", 0.0),
            # predicted values
            "pred_calories_kcal": pred.get("calories_kcal", 0.0),
            "pred_protein_g": pred.get("protein_g", 0.0),
            "pred_fat_g": pred.get("fat_g", 0.0),
            "pred_carbs_g": pred.get("carbs_g", 0.0),
            "pred_fiber_g": pred.get("fiber_g", 0.0),
            "pred_sugar_g": pred.get("sugar_g", 0.0),
            "pred_sodium_mg": pred.get("sodium_mg", 0.0),
            # per-image accuracy
            "accuracy": item.get("accuracy", 0.0),
        })
    df = pd.DataFrame(rows)
    df.to_excel(out_excel_path, index=False)
    return eval_result


# --- Conversational follow-ups (chat) ---
class NutritionChatSession:
    """Lightweight chat session that keeps the image and pipeline context for follow-ups."""

    def __init__(self, image_bytes: bytes, mime_type: str, context: Dict[str, Any], model: str = "gpt-4.1") -> None:
        self.client = OpenAIClient(model=model)
        self.image_bytes = image_bytes
        self.mime_type = mime_type or "image/jpeg"
        self.context = context  # e.g., {dish, ingredients, quantity, nutrition}
        self.history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful nutrition assistant. Use the provided image and the "
                    "structured results (dish, ingredients, quantity, nutrition) as ground truth. "
                    "Answer follow-up questions concisely and accurately."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Context: {json.dumps(self.context)}"},
                ],
            },
        ]

    def ask(self, question: str) -> str:
        import base64

        # Always include the image to anchor visual references
        b64 = base64.b64encode(self.image_bytes).decode("utf-8")
        image_url = f"data:{self.mime_type};base64,{b64}"
        self.history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
        resp = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=self.history,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        # Keep assistant response in history
        self.history.append({"role": "assistant", "content": answer})
        return answer


def create_chat_session_from_result(result: Dict[str, Any]) -> NutritionChatSession:
    """Build a chat session using the final pipeline result for follow-ups about the same image."""
    image_bytes = (result or {}).get("image_bytes", b"")
    mime_type = (result or {}).get("mime_type", "image/jpeg")
    context = {
        "dish": (result or {}).get("dish", {}),
        "ingredients": (result or {}).get("ingredients", {}),
        "quantity": (result or {}).get("quantity", {}),
        "nutrition": (result or {}).get("nutrition", {}),
    }
    return NutritionChatSession(image_bytes=image_bytes, mime_type=mime_type, context=context)
