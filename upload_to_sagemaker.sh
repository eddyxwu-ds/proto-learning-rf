#!/bin/bash
# Upload your project files to SageMaker
# Run this from your project directory

echo "📁 Uploading project files to SageMaker..."

# Create a zip of your project (excluding large data files)
zip -r rf_signal_project.zip . -x "Batch_Dir_*/*" "*.pt" "*.pkl" "*.npy" "*.zip"

# Upload to S3 (you'll need to replace with your S3 bucket)
# aws s3 cp rf_signal_project.zip s3://your-bucket/rf_signal_project.zip

echo "✅ Upload complete!"
echo "📋 Next steps:"
echo "1. Open the SageMaker notebook instance"
echo "2. Download the zip file from S3"
echo "3. Extract and run your training script"
