import beam
import tempfile
import base64
import os
from typing import Dict, Any

# Use the official PaddleOCR-VL Docker image and add PaddlePaddle
image = (
    beam.Image(
        base_image="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest"
    )
    .add_commands([
        "pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    ])
)

# Global pipeline variable for model persistence
pipeline = None

def initialize_pipeline():
    """Initialize PaddleOCR-VL pipeline with GPU support"""
    global pipeline
    if pipeline is None:
        print("Initializing PaddleOCR-VL pipeline with GPU...")
        from paddleocr import PaddleOCRVL
        
        # Initialize with full capabilities
        pipeline = PaddleOCRVL(
            use_doc_orientation_classify=True,  # Document rotation correction
            use_doc_unwarping=True,            # Document perspective correction  
            use_layout_detection=True          # Layout analysis
        )
        print("PaddleOCR-VL pipeline initialized successfully!")
    return pipeline

@beam.endpoint(
    image=image,
    gpu="RTX4090",
    cpu=2,
    memory="8Gi",
    name="paddleocr-vl-extract"
)
def extract_text_and_analyze(
    image_data: str,
    output_format: str = "json",
    include_character_metrics: bool = True,
    include_layout_analysis: bool = True
) -> Dict[str, Any]:
    """
    Extract text and analyze document structure using PaddleOCR-VL
    
    Args:
        image_data: Base64 encoded image data
        output_format: 'json' or 'markdown'
        include_character_metrics: Calculate character-level metrics
        include_layout_analysis: Include layout detection results
    
    Returns:
        Comprehensive analysis results including text, structure, and metrics
    """
    try:
        # Initialize pipeline
        ocr_pipeline = initialize_pipeline()
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            temp_path = tmp_file.name
        
        try:
            # Process with PaddleOCR-VL
            print(f"Processing image with PaddleOCR-VL...")
            output = ocr_pipeline.predict(temp_path)
            
            results = []
            
            for res in output:
                # Extract structured data
                result_data = {
                    "success": True,
                    "text_content": res.text if hasattr(res, 'text') else "",
                    "structure_info": {}
                }
                
                # Add JSON structure if available
                if hasattr(res, 'json') and res.json:
                    result_data["structure_info"]["json"] = res.json
                
                # Add markdown if requested
                if output_format == "markdown" and hasattr(res, 'markdown'):
                    result_data["markdown"] = res.markdown
                
                # Add layout analysis if available and requested
                if include_layout_analysis and hasattr(res, 'layout'):
                    result_data["layout_analysis"] = res.layout
                
                # Calculate character metrics if requested
                if include_character_metrics:
                    result_data["character_metrics"] = calculate_character_metrics(res)
                
                results.append(result_data)
            
            return {
                "success": True,
                "results": results,
                "total_pages": len(results),
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "features_used": {
                        "doc_orientation_classify": True,
                        "doc_unwarping": True,
                        "layout_detection": include_layout_analysis
                    }
                }
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@beam.endpoint(
    image=image,
    gpu="RTX4090", 
    cpu=2,
    memory="4Gi",
    name="paddleocr-vl-simple"
)
def extract_text_simple(image_data: str) -> Dict[str, Any]:
    """
    Simple text extraction without layout analysis (faster)
    
    Args:
        image_data: Base64 encoded image data
    
    Returns:
        Simple text extraction results with character metrics
    """
    try:
        # Initialize pipeline
        ocr_pipeline = initialize_pipeline()
        
        # Decode image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            temp_path = tmp_file.name
        
        try:
            # Process with basic settings for speed
            output = ocr_pipeline.predict(temp_path)
            
            all_text = []
            character_metrics = []
            
            for res in output:
                if hasattr(res, 'text') and res.text:
                    all_text.append(res.text)
                    character_metrics.append(calculate_character_metrics(res))
            
            # Combine text
            full_text = "\n".join(all_text)
            words = full_text.split()
            
            # Calculate average character metrics
            if character_metrics:
                avg_metrics = {
                    "average_character_count": sum(m.get("character_count", 0) for m in character_metrics) / len(character_metrics),
                    "average_word_length": sum(m.get("average_word_length", 0) for m in character_metrics) / len(character_metrics),
                    "total_lines": sum(m.get("line_count", 0) for m in character_metrics)
                }
            else:
                avg_metrics = {"note": "No character metrics available"}
            
            return {
                "success": True,
                "extracted_text": full_text,
                "word_count": len(words),
                "character_count": len(full_text.replace(" ", "")),
                "character_metrics": avg_metrics,
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "mode": "simple_extraction"
                }
            }
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def calculate_character_metrics(ocr_result) -> Dict[str, Any]:
    """
    Calculate character-level metrics from OCR results
    """
    try:
        # Basic metrics that can be extracted from text
        text = ocr_result.text if hasattr(ocr_result, 'text') else ""
        
        if not text:
            return {"note": "No text found for character analysis"}
        
        words = text.split()
        
        return {
            "character_count": len(text.replace(" ", "")),
            "word_count": len(words),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "line_count": len(text.split('\n')),
            "note": "Character metrics from PaddleOCR-VL text analysis"
        }
        
    except Exception as e:
        return {"error": f"Character metrics calculation failed: {str(e)}"}

if __name__ == "__main__":
    # For testing
    print("PaddleOCR-VL Beam API")
    print("Deploy with:")
    print("  beam deploy app.py:extract_text_and_analyze")
    print("  beam deploy app.py:extract_text_simple")
