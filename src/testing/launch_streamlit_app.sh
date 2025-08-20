#!/usr/bin/env bash

# Friday AI Streamlit App Launcher
# Usage: ./src/testing/launch_streamlit_app.sh

set -e

echo "🎭 Launching Friday AI Streamlit Tester..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "friday_ft" ]]; then
    echo "⚠️  Please activate the conda environment first:"
    echo "   conda activate friday_ft"
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found. Please create it with your AWS credentials and HF_TOKEN"
    exit 1
fi

# Source environment variables
echo "🔧 Loading environment variables..."
source .env

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit dependencies..."
    pip install -r src/testing/requirements_streamlit.txt
fi

# Set environment variables for the app
export SAGEMAKER_ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME:-friday-rt}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "🚀 Starting Streamlit app..."
echo "   Endpoint: $SAGEMAKER_ENDPOINT_NAME"
echo "   Region: $AWS_DEFAULT_REGION"
echo ""
echo "📱 The app will open in your browser automatically"
echo "   URL: http://localhost:8501"
echo ""
echo "🛑 To stop the app, press Ctrl+C"
echo ""

# Launch Streamlit
streamlit run src/testing/friday_streamlit_tester.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base light
