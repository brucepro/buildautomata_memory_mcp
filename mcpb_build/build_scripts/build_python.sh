#!/bin/bash
# BuildAutomata Memory MCP - Python Extension Build Script

set -e  # Exit on error

echo "🏗️  Building BuildAutomata Memory Python Extension (.mcpb)"
echo ""

# Navigate to mcpb_build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BUILD_DIR"

echo "📁 Build directory: $BUILD_DIR"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/*.mcpb
echo "✅ Clean complete"
echo ""

# Verify source files exist
echo "🔍 Verifying source files..."
if [ ! -f "src/buildautomata_memory_mcp.py" ]; then
    echo "❌ Error: src/buildautomata_memory_mcp.py not found"
    exit 1
fi
echo "✅ Source files verified"
echo ""

# Check if mcpb CLI is installed
if ! command -v mcpb &> /dev/null; then
    echo "⚠️  mcpb CLI not found. Installing..."
    npm install -g @anthropic-ai/mcpb
    echo "✅ mcpb CLI installed"
fi
echo ""

# Validate manifest.json
echo "📋 Validating manifest.json..."
if [ ! -f "manifest.json" ]; then
    echo "❌ Error: manifest.json not found"
    exit 1
fi

# Check manifest is valid JSON
if ! python3 -m json.tool manifest.json > /dev/null 2>&1; then
    echo "❌ Error: manifest.json is invalid JSON"
    exit 1
fi
echo "✅ Manifest validated"
echo ""

# Create placeholder icon if it doesn't exist
if [ ! -f "icon.png" ]; then
    echo "⚠️  No icon.png found, creating placeholder..."
    # Create a simple 256x256 placeholder icon
    convert -size 256x256 xc:#4A90E2 \
            -gravity center \
            -pointsize 72 \
            -fill white \
            -annotate +0+0 "BA" \
            icon.png 2>/dev/null || echo "⚠️  ImageMagick not available, skipping icon creation"
fi
echo ""

# Package the extension
echo "📦 Packaging Python extension..."
mcpb pack --output dist/buildautomata-memory-dev.mcpb

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📦 Output: dist/buildautomata-memory-dev.mcpb"
    echo "📏 Size: $(du -h dist/buildautomata-memory-dev.mcpb | cut -f1)"
    echo ""
    echo "🚀 To install:"
    echo "   1. Open Claude Desktop"
    echo "   2. Go to Settings → Extensions"
    echo "   3. Drag dist/buildautomata-memory-dev.mcpb into the window"
    echo "   4. Click 'Install'"
    echo ""
    echo "⚠️  Note: This is the Python version. Users need Python 3.9+ installed."
    echo "   For a standalone binary version, run: build_scripts/build_binary.sh"
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
