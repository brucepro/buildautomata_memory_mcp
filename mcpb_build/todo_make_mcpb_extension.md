# BuildAutomata Memory MCP - Desktop Extension (.mcpb) TODO

## Context

Desktop Extensions (.mcpb) make MCP servers installable via one-click in Claude Desktop. Three packaging options exist: Node.js (bundled runtime), Python (user-installed runtime), or Binary (standalone executable).

## Key Research Finding (2025-11-17) ✅

**Qdrant embedded mode confirmed working!** No need to bundle Qdrant server binaries or manage processes.

```python
from qdrant_client import QdrantClient

# Embedded mode - works like SQLite
client = QdrantClient(path="./qdrant_data")  # Persistent local storage
client = QdrantClient(":memory:")            # In-memory mode (testing)
```

**Impact:** Massively simplifies .mcpb packaging. Qdrant becomes a pure Python dependency bundled with the extension. No cross-platform binary compilation needed for Qdrant (still needed for Python→binary compilation if doing standalone executable distribution).

## Key Challenge: Qdrant Dependency

BuildAutomata Memory MCP requires Qdrant vector database. Options:

### 1. Embedded Qdrant
- Use **qdrant-client with in-process mode** (if available)
- Bundle Qdrant as embedded database (SQLite-style)
- Pro: Single package, no external services
- Con: Need to verify qdrant-client supports embedded mode

### 2. Bundled Qdrant Server
- Package Qdrant binary with extension
- Auto-start Qdrant on localhost:6333 when MCP server starts
- Pro: Full Qdrant features, no user setup
- Con: Larger package size, process management complexity

### 3. Embedded Qdrant (RECOMMENDED) ✅
- **qdrant-client supports embedded mode** via `QdrantClient(path="./qdrant_data")`
- No separate Qdrant server process needed
- Works exactly like SQLite - pure Python library with local storage
- Pro: Zero process management, smaller package, simple deployment
- Con: None significant - full Qdrant features available
- **Research completed 2025-11-17:** Verified working, 768KB storage overhead

### 4. Optional Qdrant
- Make Qdrant optional, fallback to SQLite FTS only
- User can install Qdrant separately if they want vector search
- Pro: Smaller package, simpler deployment
- Con: Reduced functionality without Qdrant

### 5. Qdrant Cloud
- Use Qdrant Cloud API instead of local instance
- User provides API key in extension config
- Pro: No bundling needed, scalable
- Con: Requires internet, data leaves local machine (defeats MCP local philosophy)

## Packaging Strategies

### Strategy A: Python Extension (Tinkerers)
**Target:** Technical users who want to modify/extend functionality

**Structure:**
```
buildautomata_memory.mcpb
├── manifest.json
├── server/
│   ├── buildautomata_memory_mcp.py
│   ├── interactive_memory.py
│   └── models.py (after refactor)
├── lib/
│   ├── qdrant_client/
│   ├── sentence_transformers/
│   └── mcp/
├── qdrant/  (bundled Qdrant binary)
│   ├── qdrant (Linux/Mac)
│   └── qdrant.exe (Windows)
├── requirements.txt
└── icon.png
```

**Pros:**
- Users can read/modify source
- Easier to debug/extend
- Smaller package (no compilation overhead)

**Cons:**
- User must have Python installed
- More complex startup (launch Qdrant, then Python server)

**Manifest config:**
```json
{
  "server": {
    "type": "python",
    "entry_point": "server/buildautomata_memory_mcp.py",
    "mcp_config": {
      "command": "python",
      "args": ["${__dirname}/server/buildautomata_memory_mcp.py"]
    }
  }
}
```

### Strategy B: Binary Extension (End Users)
**Target:** Non-technical users who want zero-setup installation

**Structure:**
```
buildautomata_memory.mcpb
├── manifest.json
├── bin/
│   ├── buildautomata_memory_mcp (Mac/Linux)
│   └── buildautomata_memory_mcp.exe (Windows)
├── qdrant/
│   ├── qdrant (Mac/Linux)
│   └── qdrant.exe (Windows)
└── icon.png
```

**Pros:**
- Zero external dependencies
- Clean user experience
- Single-click install

**Cons:**
- Large package size (50-200MB with all dependencies)
- Must compile for each platform (Windows, Mac Intel, Mac ARM, Linux)
- Users can't inspect/modify code

**Build process:**
- PyInstaller: `pyinstaller --onefile buildautomata_memory_mcp.py`
- Nuitka: Better performance, smaller binaries
- Bundle platform-specific Qdrant binaries

**Manifest config:**
```json
{
  "server": {
    "type": "binary",
    "entry_point": "bin/buildautomata_memory_mcp",
    "mcp_config": {
      "command": "${__dirname}/bin/buildautomata_memory_mcp",
      "platforms": {
        "win32": {
          "command": "${__dirname}/bin/buildautomata_memory_mcp.exe"
        }
      }
    }
  }
}
```

### Strategy C: Dual Distribution
**Recommended:** Offer both packages

- `buildautomata_memory_dev.mcpb` - Python version for developers
- `buildautomata_memory.mcpb` - Binary version for end users

Users choose based on their needs.

## Qdrant Integration Approaches

### Approach 1: Auto-managed Qdrant Process
Extension manages Qdrant lifecycle:

```python
import subprocess
import atexit

def start_qdrant():
    qdrant_path = os.path.join(os.path.dirname(__file__), "qdrant", "qdrant")
    process = subprocess.Popen([qdrant_path, "--storage-path", "./qdrant_data"])
    atexit.register(lambda: process.terminate())
    time.sleep(2)  # Wait for Qdrant to start
    return process
```

### Approach 2: System Qdrant Detection
Check if Qdrant already running:

```python
def ensure_qdrant():
    try:
        # Try to connect to existing Qdrant
        client = QdrantClient("localhost", port=6333)
        client.get_collections()
        return client
    except:
        # Start bundled Qdrant if not found
        start_qdrant()
        return QdrantClient("localhost", port=6333)
```

### Approach 3: Embedded Mode (if available)
Research: Does qdrant-client support embedded/in-process mode?

```python
# Hypothetical embedded mode
client = QdrantClient(path="./qdrant_data", mode="embedded")
```

**TODO:** Investigate qdrant-client embedded capabilities

## User Configuration

Extension should collect:

```json
"user_config": {
  "storage_path": {
    "type": "directory",
    "title": "Memory Storage Path",
    "description": "Where to store memory database",
    "default": "${HOME}/.buildautomata/memory",
    "required": true
  },
  "max_memories": {
    "type": "number",
    "title": "Maximum Memories",
    "description": "Maximum number of memories to retain",
    "default": 10000,
    "min": 100,
    "max": 1000000
  },
  "enable_vector_search": {
    "type": "boolean",
    "title": "Enable Vector Search (Qdrant)",
    "description": "Enable semantic search using Qdrant. Requires more resources.",
    "default": true
  }
}
```

## Platform-Specific Considerations

### Windows
- Qdrant binary: `qdrant.exe`
- Python binary: `buildautomata_memory_mcp.exe`
- Storage path: `%APPDATA%\BuildAutomata\memory`

### macOS
- Universal binary for Apple Silicon + Intel
- Or separate packages: `buildautomata_memory_arm64.mcpb`, `buildautomata_memory_x64.mcpb`
- Qdrant binary: `qdrant` (check architecture)
- Storage path: `~/Library/Application Support/BuildAutomata/memory`

### Linux
- Multiple distro binaries or AppImage
- Qdrant binary: `qdrant`
- Storage path: `~/.local/share/buildautomata/memory`

## Build Toolchain

### For Python Extension
```bash
# Install MCPB tools
npm install -g @anthropic-ai/mcpb

# Initialize manifest
mcpb init

# Edit manifest.json with proper config

# Download Qdrant binaries for all platforms
# Windows: https://github.com/qdrant/qdrant/releases
# Mac: brew install qdrant (then find binary)
# Linux: wget qdrant release

# Bundle Python dependencies
pip install -r requirements.txt -t lib/

# Package extension
mcpb pack
```

### For Binary Extension
```bash
# Install PyInstaller or Nuitka
pip install pyinstaller

# Build for current platform
pyinstaller --onefile \
  --add-data "qdrant:qdrant" \
  buildautomata_memory_mcp.py

# Or use Nuitka (better performance)
nuitka --standalone \
  --onefile \
  --include-data-dir=qdrant=qdrant \
  buildautomata_memory_mcp.py

# Cross-compile for other platforms (complex)
# May need separate build environments (Docker, VMs, CI/CD)

# Package with MCPB tools
mcpb pack
```

## Testing Plan

1. **Local Testing:**
   - Build .mcpb package
   - Drag into Claude Desktop Settings
   - Verify installation UI shows correct info
   - Install and test MCP tools

2. **Cross-Platform Testing:**
   - Test on Windows 10/11
   - Test on macOS (Intel + ARM)
   - Test on Linux (Ubuntu, Fedora)

3. **Qdrant Integration Testing:**
   - Verify Qdrant starts automatically
   - Test vector search functionality
   - Test graceful fallback if Qdrant fails
   - Verify Qdrant shuts down on extension uninstall

4. **User Config Testing:**
   - Test storage path customization
   - Test max_memories limits
   - Test sensitive value storage (if using API keys later)

## Distribution

### Direct Distribution
- Host .mcpb files on GitHub Releases
- Users download and install manually

### Anthropic Extension Directory
- Submit to official directory
- Review process required
- One-click install from Claude Desktop

### Private Enterprise Distribution
- Host on internal servers
- Deploy via Group Policy (Windows) or MDM (macOS)
- Pre-install approved extensions

## Open Questions

1. ~~**Qdrant embedding:** Does qdrant-client support embedded/in-process mode?~~ ✅ **ANSWERED:** Yes, use `QdrantClient(path="./qdrant_data")` - works like SQLite
2. **Package size limits:** What's acceptable .mcpb size for directory submission?
3. **Update mechanism:** How do users get updates? Auto-update or manual?
4. **Multi-user:** How does extension handle multiple OS users on same machine?
5. ~~**Resource limits:** Should extension limit memory/CPU usage of Qdrant?~~ ✅ **N/A:** Embedded mode runs in-process, no separate resource management needed
6. ~~**Sentence transformers:** Bundle models or download on first run?~~ ✅ **RESEARCHED:** all-MiniLM-L6-v2 is 90MB. Decision: Download on first run to keep package size small (~10MB vs ~100MB)
7. **Cross-compilation:** Use CI/CD (GitHub Actions) to build all platform binaries?
8. **Performance on typical hardware:** Benchmarks run on i9-12900 + 64GB DDR5. Need to test on typical user hardware (16GB RAM, mid-range CPU) to verify embedded mode performance is acceptable

## Next Steps

1. **Research Phase:**
   - [x] Investigate qdrant-client embedded mode capabilities ✅ **COMPLETE** - Embedded mode confirmed working
   - [x] Check sentence-transformers model bundling size ✅ **COMPLETE** - 90MB model, download on first run
   - [ ] Determine acceptable package sizes for extension directory
   - ~~[ ] Review Qdrant binary sizes across platforms~~ ✅ **N/A** - Using embedded mode, no binaries needed
   - [ ] Test performance on typical hardware (16GB RAM, mid-range CPU) vs current benchmarks (64GB DDR5, i9-12900)

2. **Refactor Phase:**
   - [ ] Complete modular refactor (separate concerns)
   - [ ] Audit all Qdrant API calls (already started - fixed query_points bug)
   - [ ] Ensure clean shutdown/cleanup of resources
   - ~~[ ] Add proper process management for Qdrant~~ ✅ **N/A** - Embedded mode, no process management needed

3. **Prototype Phase:**
   - [ ] Build Python extension prototype
   - [ ] Test on current development machine
   - ~~[ ] Verify Qdrant auto-start works~~ ✅ **N/A** - Embedded mode initializes via `QdrantClient(path=...)`
   - [ ] Test in Claude Desktop locally

4. **Binary Phase:**
   - [ ] Compile with PyInstaller/Nuitka
   - [ ] Test binary on clean machine (no Python)
   - [ ] Optimize binary size
   - [ ] Set up cross-platform builds

5. **Polish Phase:**
   - [ ] Create proper icon.png
   - [ ] Write clear user documentation
   - [ ] Add screenshots for directory submission
   - [ ] Write CHANGELOG.md

6. **Distribution Phase:**
   - [ ] Test installation flow end-to-end
   - [ ] Submit to Anthropic extension directory
   - [ ] Create GitHub Release with binaries
   - [ ] Announce to community

## Resources

- [MCPB Specification](https://github.com/anthropics/mcpb/blob/main/README.md)
- [Manifest Documentation](https://github.com/anthropics/mcpb/blob/main/MANIFEST.md)
- [MCPB Examples](https://github.com/anthropics/mcpb/tree/main/examples)
- [Qdrant Releases](https://github.com/qdrant/qdrant/releases)
- [PyInstaller Docs](https://pyinstaller.org/)
- [Nuitka Docs](https://nuitka.net/)

## Decision Log

**2025-11-17 (Initial):** Identified need for .mcpb packaging. Main blocker: Qdrant bundling strategy. Two paths forward: Python (tinkerers) + Binary (end users) dual distribution recommended.

**2025-11-17 (Research Complete):**
- ✅ Qdrant embedded mode confirmed - no server binary needed
- ✅ sentence-transformers model: 90MB, download on first run
- ✅ Performance at 1100 memories: 3.5ms queries (i9-12900 + 64GB DDR5)
- **Primary strategy: Binary extension** - Zero user setup, one-click install
- **Secondary: Python extension** - For developers who want to modify code
- **Blocker removed:** Embedded Qdrant eliminates process management complexity
- **Next:** Wait for Claude Code Web audit/refactor, then build binary with PyInstaller/Nuitka
- **Test plan:** Compile binary, test on clean Windows VM (no dev tools), then test on i5 laptop (32GB DDR4) for real-world performance baseline
