import beam
import tempfile
import base64
import os
from typing import Dict, Any, Optional

mount_path = "./protocols"

# Use the official PaddleOCR-VL Docker image and add PaddlePaddle with model caching
image = (
    beam.Image(
        base_image="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest"
    )
    .add_commands([
        "pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    ])
)

# Create persistent volume for model caching at PaddleOCR's actual model directory
model_cache = beam.Volume(name="paddleocr-models", mount_path="/home/paddleocr/.paddlex/official_models")

# Cloudflare R2 bucket for file uploads (S3-compatible, supported by Beam)
uploads_bucket = beam.CloudBucket(
    name="protocols",  # Must match your actual R2 bucket name
    mount_path=mount_path,
    config=beam.CloudBucketConfig(
        access_key="BEAM_S3_KEY", 
        secret_key="BEAM_S3_SECRET",
        endpoint="https://50e1f4714be505bee485af31b51492f1.r2.cloudflarestorage.com",  # Hardcoded for testing
        region="auto"  # R2 uses "auto" region
    )
)

# Global pipeline variable for model persistence
pipeline = None

def initialize_pipeline():
    """Initialize PaddleOCR-VL pipeline with GPU support and model caching"""
    global pipeline
    if pipeline is None:
        print("Initializing PaddleOCR-VL pipeline with GPU and model caching...")
        
        from paddleocr import PaddleOCRVL
        
        # Initialize with full capabilities - models will be cached in mounted volume
        pipeline = PaddleOCRVL(
            use_doc_orientation_classify=True,  # Document rotation correction
            use_doc_unwarping=True,            # Document perspective correction  
            use_layout_detection=True          # Layout analysis
        )
        print("PaddleOCR-VL pipeline initialized successfully with cached models!")
    return pipeline

def prepare_input_file(image_data: Optional[str] = None, file_name: Optional[str] = None) -> str:
    """
    Prepare input file from either base64 data or S3 file upload
    
    Args:
        image_data: Base64 encoded image data (optional)
        file_name: Name of file uploaded to S3 bucket (optional)
    
    Returns:
        Path to temporary file ready for processing
        
    Raises:
        ValueError: If neither or both parameters are provided
    """
    if not image_data and not file_name:
        raise ValueError("Either image_data or file_name must be provided")
    
    if image_data and file_name:
        raise ValueError("Provide either image_data OR file_name, not both")
    
    if image_data:
        # Handle base64 input (existing method)
        if image_data.startswith('data:image') or image_data.startswith('data:application'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            return tmp_file.name
    
    elif file_name:
        # Handle S3 file upload (new method)
        file_path = os.path.join(mount_path, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found in uploads: {file_name}")
        
        return file_path
    
@beam.endpoint(
    image=image,
    gpu="RTX4090",
    cpu=2,
    memory="8Gi",
    volumes=[model_cache, uploads_bucket],
    name="paddleocr-vl-extract"
)
def extract_text_and_analyze(
    image_data: Optional[str] = None,
    file_name: Optional[str] = None,
    output_format: str = "json",
    include_character_metrics: bool = True,
    include_layout_analysis: bool = True
) -> Dict[str, Any]:
    """
    Extract text and analyze document structure using PaddleOCR-VL
    
    Supports two input methods:
    1. Base64 encoded image/PDF data via image_data parameter
    2. File upload to S3 bucket via file_name parameter
    
    Args:
        image_data: Base64 encoded image/PDF data (optional)
        file_name: Name of file uploaded to S3 bucket (optional)
        output_format: 'json' or 'markdown'
        include_character_metrics: Calculate character-level metrics
        include_layout_analysis: Include layout detection results
    
    Returns:
        Comprehensive analysis results including text, structure, and metrics
    """
    try:
        # Initialize pipeline
        ocr_pipeline = initialize_pipeline()
        
        # Prepare input file from either base64 or S3 upload
        input_path = prepare_input_file(image_data, file_name)
        temp_file_created = image_data is not None  # Only delete if we created it
        
        try:
            # Process with PaddleOCR-VL
            print(f"Processing document with PaddleOCR-VL: {input_path}")
            output = ocr_pipeline.predict(input_path)
            
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
                "input_method": "base64" if image_data else "s3_upload",
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
            # Clean up temporary file only if we created it
            if temp_file_created and os.path.exists(input_path):
                os.unlink(input_path)
                
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
    volumes=[model_cache, uploads_bucket],
    name="paddleocr-vl-simple"
)
def extract_text_simple(
    image_data: Optional[str] = None,
    file_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simple text extraction without layout analysis (faster)
    
    Supports two input methods:
    1. Base64 encoded image/PDF data via image_data parameter
    2. File upload to S3 bucket via file_name parameter
    
    Args:
        image_data: Base64 encoded image/PDF data (optional)
        file_name: Name of file uploaded to S3 bucket (optional)
    
    Returns:
        Simple text extraction results with character metrics
    """
    try:
        # Initialize pipeline
        ocr_pipeline = initialize_pipeline()
        
        # Prepare input file from either base64 or S3 upload
        input_path = prepare_input_file(image_data, file_name)
        temp_file_created = image_data is not None
        
        try:
            # Process with basic settings for speed
            print(f"Processing document with PaddleOCR-VL (simple): {input_path}")
            output = ocr_pipeline.predict(input_path)
            
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
                "input_method": "base64" if image_data else "s3_upload",
                "processing_info": {
                    "model": "PaddleOCR-VL",
                    "gpu_accelerated": True,
                    "mode": "simple_extraction"
                }
            }
            
        finally:
            # Clean up temporary file only if we created it
            if temp_file_created and os.path.exists(input_path):
                os.unlink(input_path)
                
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
    print("PaddleOCR-VL Beam API with Dual Input Support")
    print("Deploy with:")
    print("  beam deploy app.py:extract_text_and_analyze")
    print("  beam deploy app.py:extract_text_simple")
    print("")
    print("Usage:")
    print("  Base64: {\"image_data\": \"data:image/jpeg;base64,...\"}")
    print("  S3 Upload: {\"file_name\": \"document.pdf\"}")
