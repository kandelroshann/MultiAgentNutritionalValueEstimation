from deepeval.metrics import BaseMetric

class NumericAccuracy(BaseMetric):
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def measure(self, reference, prediction):
        # If input is a list (like the JSON list with one item)
        if isinstance(reference, list) and isinstance(prediction, list):
            if len(reference) == len(prediction) == 1:
                reference = reference[0]
                prediction = prediction[0]
            else:
                scores = [
                    self.measure(ref, pred)
                    for ref, pred in zip(reference, prediction)
                ]
                return sum(scores) / len(scores)

        # Handle dicts (JSON-style objects)
        if isinstance(reference, dict) and isinstance(prediction, dict):
            numeric_fields = [
                "calories_kcal",
                "protein_g",
                "fat_g",
                "carbs_g",
                "fiber_g",
                "sugar_g",
                "sodium_mg",
            ]
            scores = []
            for field in numeric_fields:
                ref_val = reference.get(field)
                pred_val = prediction.get(field)
                try:
                    ref_val, pred_val = float(ref_val), float(pred_val)
                    rel_err = abs(ref_val - pred_val) / abs(ref_val) if ref_val else 0
                    score = (
                        1.0
                        if rel_err <= self.tolerance
                        else max(0.0, 1 - rel_err / self.tolerance)
                    )
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            return sum(scores) / len(scores) if scores else 0.0

        # Handle scalars
        try:
            reference = float(reference)
            prediction = float(prediction)
            if reference == 0:
                return 1.0 if prediction == 0 else 0.0
            rel_err = abs(reference - prediction) / abs(reference)
            return (
                1.0
                if rel_err <= self.tolerance
                else max(0.0, 1 - rel_err / self.tolerance)
            )
        except Exception:
            return 0.0

    def __str__(self):
        return f"NumericAccuracy(tolerance={self.tolerance})"



# Example: comparing calorie predictions
metric = NumericAccuracy(tolerance=0.1)  # 10% tolerance

score = metric.measure(
    reference=[
  {
    "image_name": "fish_and_chips_1001881.jpg",
    "dish_name": "fish and chips",
    "calories_kcal": 875,
    "protein_g": 34,
    "fat_g": 42,
    "carbs_g": 101,
    "fiber_g": 5,
    "sugar_g": 12,
    "sodium_mg": 14
  }],      # ground truth
    prediction=[
  {
    "image_name": "fish_and_chips_1001881.jpg",
    "dish_name": "fish and chips",
    "calories_kcal": 811,
    "protein_g": 34,
    "fat_g": 40,
    "carbs_g": 100,
    "fiber_g": 5,
    "sugar_g": 12,
    "sodium_mg": 11
  }]    # model output
)
print("Numeric Accuracy:", score)