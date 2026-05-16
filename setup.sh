#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
ZIP_FILE="vosk-model-small-en-us-0.15.zip"
EXTRACTED_DIR="vosk-model-small-en-us-0.15"
TARGET_DIR="model_en"

if [ -d "$TARGET_DIR" ]; then
    echo "✅ $TARGET_DIR already exists. Nothing to do."
    exit 0
fi

# Prefer wget, fallback to curl
downloader=""
if command -v wget &> /dev/null; then
    downloader="wget -q --show-progress"
elif command -v curl &> /dev/null; then
    downloader="curl -L -o $ZIP_FILE --progress-bar"
else
    echo "❌ Error: neither wget nor curl is installed. Please install one of them."
    exit 1
fi

echo "⬇️  Downloading Vosk wake-word model..."
$downloader "$MODEL_URL"

echo "📦 Extracting..."
unzip -q "$ZIP_FILE"

echo "📁 Renaming to $TARGET_DIR..."
mv "$EXTRACTED_DIR" "$TARGET_DIR"

echo "🧹 Cleaning up..."
rm -f "$ZIP_FILE"

echo "✅ Done. $TARGET_DIR is ready."
