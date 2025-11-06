# S3 Setup Instructions

To enable file upload support via S3, you need to configure Beam secrets for your S3 credentials.

## Prerequisites

1. **AWS S3 Bucket** - Create a bucket for file uploads
2. **AWS Access Keys** - Get access key ID and secret access key with S3 permissions

## Setup Commands

```bash
# Set your S3 access credentials as Beam secrets
beam secret create BEAM_S3_KEY "your-aws-access-key-id"
beam secret create BEAM_S3_SECRET "your-aws-secret-access-key"

# Update the bucket name in app.py if needed
# Change "paddleocr-uploads" to your actual bucket name
```

## Bucket Configuration

Update the bucket name in `app.py`:

```python
uploads_bucket = beam.CloudBucket(
    name="your-actual-bucket-name",  # Change this to your bucket
    mount_path="/uploads",
    config=beam.CloudBucketConfig(
        access_key="BEAM_S3_KEY", 
        secret_key="BEAM_S3_SECRET"
    )
)
```

## Usage Examples

### Upload File to S3
```bash
# Upload a PDF to your S3 bucket
aws s3 cp document.pdf s3://your-bucket-name/

# Upload an image to your S3 bucket  
aws s3 cp image.jpg s3://your-bucket-name/
```

### Call API with S3 File
```bash
curl -X POST "https://your-beam-endpoint" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{"file_name": "document.pdf"}'
```

### Call API with Base64 (Still Supported)
```bash
curl -X POST "https://your-beam-endpoint" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{"image_data": "data:image/jpeg;base64,/9j4AAQ..."}'
```

## Benefits

- **Large files**: Handle PDFs and images >20MB
- **Direct upload**: No base64 encoding required
- **Multi-page PDFs**: Full document processing
- **Better performance**: No encoding/decoding overhead
- **Batch processing**: Upload multiple files, process sequentially

## Security Note

Make sure your S3 bucket has appropriate permissions and your AWS keys have minimal required permissions (read/write to the specific bucket only).
