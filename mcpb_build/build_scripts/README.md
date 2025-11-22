# Build Scripts

These scripts build the BuildAutomata Memory MCP extension in different formats.

## Prerequisites

### For Python Extension (build_python.sh)
- Node.js 16+ (for mcpb CLI)
- Python 3.9+

### For Binary Extension (build_binary.sh)
- Node.js 16+ (for mcpb CLI)
- Python 3.9+
- PyInstaller (`pip install pyinstaller`)
- ~2GB free disk space for build artifacts
- ~10 minutes for first build

## Building

### Python Extension (Developers)
```bash
./build_scripts/build_python.sh
```

Output: `dist/buildautomata-memory-dev.mcpb`

**Pros:**
- Small package size (~10MB)
- Users can inspect/modify source code
- Fast build time

**Cons:**
- Requires Python 3.9+ installed on user's machine
- User must install dependencies

### Binary Extension (End Users)
```bash
./build_scripts/build_binary.sh
```

Output: `dist/buildautomata-memory.mcpb`

**Pros:**
- Zero dependencies - works standalone
- No Python installation required
- Single-click install for end users

**Cons:**
- Large package size (50-200MB depending on platform)
- Platform-specific (must build on target OS)
- Slower build time (~10 minutes)

## Testing

### Local Testing (before packaging)
```bash
# Test the MCP server directly
cd mcpb_build/src
python3 buildautomata_memory_mcp.py

# Should start MCP server on stdio
# Press Ctrl+C to stop
```

### Extension Testing (after packaging)
1. Build the extension with one of the scripts above
2. Open Claude Desktop
3. Go to Settings → Extensions
4. Drag the .mcpb file from `dist/` into the extensions window
5. Click "Install"
6. Restart Claude Desktop
7. Test memory commands in a conversation

## Platform-Specific Builds

### Windows
Run on Windows machine:
```bash
./build_scripts/build_binary.sh
# Produces: buildautomata-memory.exe
```

### macOS
Run on Mac:
```bash
./build_scripts/build_binary.sh
# Produces universal binary for Intel + Apple Silicon
```

### Linux
Run on Linux:
```bash
./build_scripts/build_binary.sh
# Produces Linux ELF binary
```

## Cross-Platform Strategy

For official releases, use CI/CD (GitHub Actions) to build all platforms:
- Windows x64
- macOS Universal (Intel + ARM)
- Linux x64
- Linux ARM64

See `.github/workflows/build.yml` (TODO) for automated builds.

## Troubleshooting

### "mcpb: command not found"
```bash
npm install -g @anthropic-ai/mcpb
```

### "pyinstaller: command not found"
```bash
pip3 install pyinstaller
```

### Build fails with import errors
Make sure all dependencies are in `requirements.txt`:
```bash
pip3 install -r requirements.txt
```

### Binary too large (>200MB)
Try using Nuitka instead of PyInstaller for smaller binaries:
```bash
pip3 install nuitka
# See build_binary.sh for Nuitka configuration
```

### Extension installs but doesn't work
1. Check Claude Desktop logs: `~/Library/Logs/Claude/` (Mac) or `%APPDATA%/Claude/logs/` (Windows)
2. Test MCP server directly: `python3 src/buildautomata_memory_mcp.py`
3. Verify manifest.json is valid JSON: `python3 -m json.tool manifest.json`

## File Size Estimates

| Package Type | Uncompressed | Compressed (.mcpb) |
|--------------|--------------|-------------------|
| Python       | ~10 MB       | ~8 MB            |
| Binary       | ~150 MB      | ~60 MB           |

The binary includes:
- Python runtime (~40MB)
- Qdrant client libs (~30MB)
- sentence-transformers (~50MB)
- Application code (~10MB)
- Dependencies (~20MB)

## Development Workflow

1. Make changes to source in `../` (main repo)
2. Copy updated files to `src/`: `cp ../*.py src/`
3. Test locally: `python3 src/buildautomata_memory_mcp.py`
4. Build extension: `./build_scripts/build_python.sh`
5. Test in Claude Desktop
6. If working, build binary: `./build_scripts/build_binary.sh`
7. Test binary on clean machine (no dev tools)
8. Distribute both versions

## Next Steps

- [ ] Add icon.png (256x256)
- [ ] Add screenshots/ for extension directory
- [ ] Set up GitHub Actions for cross-platform builds
- [ ] Create release workflow
- [ ] Submit to Anthropic extension directory
