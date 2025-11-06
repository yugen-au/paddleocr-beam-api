# Cloudflare R2 Setup Instructions

To enable file upload support via Cloudflare R2 (S3-compatible storage), you need to configure Beam secrets for your R2 credentials.

## Prerequisites

1. **Cloudflare R2 Bucket** - Create a bucket in your Cloudflare dashboard
2. **R2 API Token** - Generate R2 API token with read/write permissions

## Setup Commands

```bash
# Set your Cloudflare R2 credentials as Beam secrets
beam secret create BEAM_S3_KEY "your-r2-access-key-id"
beam secret create BEAM_S3_SECRET "your-r2-secret-access-key"
beam secret create BEAM_R2_ENDPOINT "https://your-account-id.r2.cloudflarestorage.com"
```

## Bucket Configuration

The bucket configuration in `app.py` with proper endpoint support:

```python
uploads_bucket = beam.CloudBucket(
    name="your-r2-bucket-name",  # Must match your actual R2 bucket name
    mount_path="/uploads",
    config=beam.CloudBucketConfig(
        access_key="BEAM_S3_KEY", 
        secret_key="BEAM_S3_SECRET",
        endpoint="BEAM_R2_ENDPOINT",  # R2 endpoint URL as secret
        region="auto"  # R2 uses "auto" region
    )
)
```

**Key Points:**
- Use `endpoint` parameter (not `endpoint_url`)
- The `name` field must exactly match your R2 bucket name
- All sensitive info stored as Beam secrets

## Getting Cloudflare R2 Credentials

1. **Go to Cloudflare Dashboard** → R2 Object Storage
2. **Create a bucket** for your uploads
3. **Generate API Token:**
   - Go to "Manage R2 API Tokens"
   - Click "Create API Token"
   - Set permissions: Object Read & Write
   - Copy the Access Key ID and Secret Access Key
4. **Get your Account ID:**
   - Found in the right sidebar of your Cloudflare dashboard
   - Used in the endpoint URL

## Usage Examples

### Upload File to R2
```bash
# Using AWS CLI (works with R2)
aws s3 cp document.pdf s3://your-r2-bucket/ \
  --endpoint-url https://your-account-id.r2.cloudflarestorage.com

# Using rclone (alternative)
rclone copy document.pdf r2:your-r2-bucket/
```

### Call API with R2 File
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

## Cloudflare R2 Benefits

- **Cost-effective**: No egress fees for data retrieval
- **S3-compatible**: Works with existing S3 tools and SDKs
- **Global edge**: Fast access from Cloudflare's global network
- **Simple pricing**: Predictable storage and operation costs
- **Large files**: Handle PDFs and images >20MB
- **Multi-page PDFs**: Full document processing

## Example Configuration

Simple and clean configuration:

```python
uploads_bucket = beam.CloudBucket(
    name="paddleocr-documents",  # Your actual R2 bucket name
    mount_path="/uploads",
    config=beam.CloudBucketConfig(
        access_key="BEAM_S3_KEY", 
        secret_key="BEAM_S3_SECRET",
        region="auto"
    )
)
```

**Setup commands:**
```bash
beam secret create BEAM_S3_KEY "a1b2c3d4e5f6..."
beam secret create BEAM_S3_SECRET "xyz789abc123..."
```

## AWS CLI Configuration for R2

Configure AWS CLI to work with Cloudflare R2:

```bash
# Configure AWS CLI profile for R2
aws configure set aws_access_key_id your-r2-access-key-id --profile r2
aws configure set aws_secret_access_key your-r2-secret-access-key --profile r2
aws configure set region auto --profile r2

# Upload files using R2 profile
aws s3 cp document.pdf s3://your-r2-bucket/ \
  --endpoint-url https://your-account-id.r2.cloudflarestorage.com \
  --profile r2
```

## Security Notes

- **API Tokens**: Use R2 API tokens with minimal required permissions (read/write to specific bucket only)
- **Bucket Permissions**: Configure appropriate bucket policies and access controls
- **Account ID**: Keep your Cloudflare account ID secure as it's part of the endpoint URL
- **Token Rotation**: Regularly rotate your R2 API tokens for security

## Troubleshooting

- **Endpoint URL**: Make sure to use your specific account ID in the endpoint URL
- **Region**: Always use "auto" for Cloudflare R2
- **Bucket Names**: R2 bucket names must be globally unique
- **CORS**: Configure CORS settings if uploading directly from web browsers
