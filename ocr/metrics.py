"""Character-level metrics derived from OCR text output."""
from typing import Any, Dict


def calculate_character_metrics(ocr_result) -> Dict[str, Any]:
    """Calculate character-level metrics from an OCR result object."""
    try:
        text = ocr_result.text if hasattr(ocr_result, "text") else ""

        if not text:
            return {"note": "No text found for character analysis"}

        words = text.split()

        return {
            "character_count": len(text.replace(" ", "")),
            "word_count": len(words),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "line_count": len(text.split("\n")),
            "note": "Character metrics from PaddleOCR-VL text analysis",
        }
    except Exception as e:
        return {"error": f"Character metrics calculation failed: {str(e)}"}
