# PaddleOCR-VL Beam API

A GPU-accelerated OCR API built with PaddleOCR-VL and deployed on Beam.cloud for document text extraction and analysis.

## Features

- **GPU-accelerated processing** with RTX 4090
- **Document unwarping** and perspective correction
- **Layout detection** and structure analysis
- **Text extraction** with character metrics
- **RESTful API** endpoints for easy integration

## Architecture

- **Base**: Official PaddleOCR-VL Docker image
- **Framework**: PaddlePaddle GPU with CUDA 12.6
- **Deployment**: Beam.cloud with auto-scaling
- **GPU**: RTX 4090 for cost-effective processing

## API Endpoints

### Extract Text and Analyze (Full Analysis)
**Endpoint:** `POST /extract_text_and_analyze`

**Input Methods:**

**Method 1 - Base64 (Good for small files):**
```json
{
  "image_data": "data:image/jpeg;base64,/9j4AAQSkZJRgABA...",
  "output_format": "json",
  "include_character_metrics": true,
  "include_layout_analysis": true
}
```

**Method 2 - Cloudflare R2 Upload (Good for large files/PDFs):**
```json
{
  "file_name": "document.pdf",
  "output_format": "json", 
  "include_character_metrics": true,
  "include_layout_analysis": true
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text_content": "Extracted text...",
      "character_metrics": {
        "character_count": 245,
        "word_count": 42,
        "average_word_length": 5.8,
        "line_count": 8
      },
      "structure_info": {...},
      "layout_analysis": {...}
    }
  ],
  "processing_info": {
    "model": "PaddleOCR-VL",
    "gpu_accelerated": true
  }
}
```

### Simple Text Extraction
**Endpoint:** `POST /extract_text_simple`

**Input Methods:**

**Method 1 - Base64:**
```json
{
  "image_data": "data:image/jpeg;base64,/9j4AAQSkZJRgABA..."
}
```

**Method 2 - Cloudflare R2 Upload:**
```json
{
  "file_name": "document.pdf"
}
```

**Response:**
```json
{
  "success": true,
  "extracted_text": "Full extracted text...",
  "word_count": 42,
  "character_count": 245,
  "character_metrics": {...}
}
```

## Deployment

### Prerequisites
- [Beam CLI](https://docs.beam.cloud) installed and configured
- Beam account with API token

### Deploy to Beam
```bash
# Clone the repository
git clone https://github.com/yugen-au/paddleocr-beam-api.git
cd paddleocr-beam-api

# Deploy the endpoint
beam deploy app.py:extract_text_and_analyze
```

### Alternative: Deploy Simple Extraction
```bash
beam deploy app.py:extract_text_simple
```

## Local Development

The functions can be tested locally using Beam's serve command:

```bash
# Start development server
beam serve app.py:extract_text_and_analyze

# Test with curl
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"image_data": "data:image/jpeg;base64,..."}'
```

## Configuration

### GPU Settings
- **GPU Type**: RTX 4090 (24GB VRAM)
- **Memory**: 8GB for full analysis, 4GB for simple extraction
- **CPU**: 2 cores
- **Auto-scaling**: 0-1 instances based on demand

### Model Configuration
- **Document Orientation**: Enabled
- **Document Unwarping**: Enabled  
- **Layout Detection**: Configurable
- **Character Metrics**: Configurable

## Cost Optimization

- **RTX 4090**: $0.000192/second (~$0.69/hour)
- **Auto-scaling**: Spins down after inactivity
- **Efficient processing**: GPU acceleration reduces processing time

Example costs:
- **100 requests/day**: ~$2-3/month
- **1000 requests/day**: ~$15-20/month

## Integration Examples

### n8n Workflow
Use the HTTP Request node with:
- **Method**: POST
- **URL**: Your deployed Beam endpoint
- **Body**: JSON with base64 image data

### Python
```python
import requests
import base64

# Encode image
with open("document.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Call API
response = requests.post(
    "https://your-beam-endpoint",
    json={
        "image_data": f"data:image/jpeg;base64,{image_data}",
        "include_character_metrics": True
    }
)

result = response.json()
print(result["results"][0]["character_metrics"])
```

## Troubleshooting

### Common Issues
- **"No module named 'paddle'"**: Ensure PaddlePaddle is installed in the image
- **OpenGL errors**: The official PaddleOCR image resolves most OpenCV/OpenGL issues
- **Memory errors**: Increase memory allocation for large documents

### Performance Tips
- Use `extract_text_simple` for faster processing when layout analysis isn't needed
- Optimize image size before sending (PaddleOCR works well with compressed images)
- Consider batch processing for multiple documents

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Beam serve
5. Submit a pull request

## Support

For issues related to:
- **PaddleOCR-VL**: Check [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR)
- **Beam deployment**: Check [Beam documentation](https://docs.beam.cloud)
- **This implementation**: Open an issue in this repository
