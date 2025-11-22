#!/bin/bash
# BuildAutomata Memory MCP - Binary Extension Build Script
# Creates standalone executable with PyInstaller (no Python required)

set -e  # Exit on error

echo "🏗️  Building BuildAutomata Memory Binary Extension (.mcpb)"
echo ""

# Navigate to mcpb_build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BUILD_DIR"

echo "📁 Build directory: $BUILD_DIR"
echo ""

# Detect platform
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

echo "🖥️  Platform: $PLATFORM ($ARCH)"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/*.mcpb
echo "✅ Clean complete"
echo ""

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "⚠️  PyInstaller not found. Installing..."
    pip3 install pyinstaller
    echo "✅ PyInstaller installed"
fi
echo ""

# Install dependencies in virtual environment for bundling
echo "📦 Installing dependencies..."
rm -rf venv_build
python3 -m venv venv_build
source venv_build/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q pyinstaller
echo "✅ Dependencies installed"
echo ""

# Build binary with PyInstaller
echo "🔨 Compiling binary with PyInstaller..."
echo "   This may take 5-10 minutes on first run..."
echo ""

pyinstaller --onefile \
    --name buildautomata-memory \
    --add-data "src:src" \
    --hidden-import qdrant_client \
    --hidden-import sentence_transformers \
    --hidden-import sklearn \
    --hidden-import torch \
    --hidden-import transformers \
    --hidden-import numpy \
    --hidden-import mcp \
    --collect-all qdrant_client \
    --collect-all sentence_transformers \
    --noconfirm \
    --clean \
    src/buildautomata_memory_mcp.py

echo ""
echo "✅ Binary compilation complete"
echo ""

# Deactivate virtual environment
deactivate

# Create binary manifest (different from Python version)
echo "📋 Creating binary manifest..."
cat > manifest_binary.json << 'EOF'
{
  "name": "buildautomata-memory",
  "version": "1.1.0",
  "description": "Persistent episodic memory system for Claude with temporal versioning, semantic search, and graph navigation. Standalone binary - no Python required.",
  "author": "Jurgen Bruce",
  "license": "MIT",
  "homepage": "https://github.com/brucepro/claudecode_playground",
  "icon": "icon.png",

  "server": {
    "type": "binary",
    "mcp_config": {
      "command": "${__dirname}/bin/buildautomata-memory",
      "platforms": {
        "win32": {
          "command": "${__dirname}/bin/buildautomata-memory.exe"
        },
        "darwin": {
          "command": "${__dirname}/bin/buildautomata-memory"
        },
        "linux": {
          "command": "${__dirname}/bin/buildautomata-memory"
        }
      }
    }
  },

  "user_config": {
    "username": {
      "type": "string",
      "title": "Username",
      "description": "Your username for organizing memories",
      "default": "${USER}",
      "required": false
    },
    "agent_name": {
      "type": "string",
      "title": "Agent Name",
      "description": "Name for this Claude instance",
      "default": "desktop",
      "required": false
    },
    "max_memories": {
      "type": "number",
      "title": "Maximum Memories",
      "description": "Maximum number of memories to retain",
      "default": 10000,
      "min": 100,
      "max": 1000000,
      "required": false
    }
  },

  "platforms": {
    "win32": {"supported": true},
    "darwin": {"supported": true},
    "linux": {"supported": true}
  },

  "permissions": {
    "filesystem": {
      "read": true,
      "write": true,
      "paths": [
        "${HOME}/.buildautomata",
        "${HOME}/Library/Application Support/BuildAutomata",
        "${APPDATA}/BuildAutomata"
      ]
    },
    "network": {
      "required": true,
      "reason": "Download embedding model on first run (90MB, one-time)"
    }
  },

  "tags": [
    "memory",
    "persistence",
    "semantic-search",
    "standalone",
    "binary"
  ]
}
EOF
echo "✅ Binary manifest created"
echo ""

# Create bin directory and copy binary
echo "📦 Organizing binary package..."
mkdir -p bin
cp dist/buildautomata-memory bin/

# Rename to match manifest
mv manifest.json manifest_python.json.bak
mv manifest_binary.json manifest.json
echo "✅ Package organized"
echo ""

# Package with mcpb
echo "📦 Creating .mcpb package..."
if ! command -v mcpb &> /dev/null; then
    echo "⚠️  mcpb CLI not found. Installing..."
    npm install -g @anthropic-ai/mcpb
fi

mcpb pack --output dist/buildautomata-memory.mcpb

# Restore original manifest
mv manifest.json manifest_binary.json.bak
mv manifest_python.json.bak manifest.json

echo ""
echo "✅ Build successful!"
echo ""
echo "📦 Output: dist/buildautomata-memory.mcpb"
echo "📏 Size: $(du -h dist/buildautomata-memory.mcpb | cut -f1)"
echo "💾 Binary: $(du -h bin/buildautomata-memory | cut -f1)"
echo ""
echo "🚀 To install:"
echo "   1. Open Claude Desktop"
echo "   2. Go to Settings → Extensions"
echo "   3. Drag dist/buildautomata-memory.mcpb into the window"
echo "   4. Click 'Install'"
echo ""
echo "✅ This is a standalone binary - NO Python installation required!"
echo "⚠️  Note: Binary is platform-specific. Build on target OS for best results."
echo ""
echo "🖥️  Current build: $PLATFORM-$ARCH"

# Cleanup
rm -rf venv_build
echo ""
echo "🧹 Cleanup complete"
