# BuildAutomata Memory MCP - Setup Guide

## Installation Options

### Option 1: Easy Install (Gumroad Bundle) 💰
**Best for: Non-technical users, Windows users, commercial use**

1. Purchase at [Gumroad](https://brucepro1.gumroad.com/l/zizjl)
2. Extract the ZIP file
3. Double-click `start_qdrant.bat`
4. Configure Claude Desktop (see below)
5. Done! Everything pre-configured.

**Includes:**
- Pre-compiled Qdrant server (no Docker needed)
- One-click startup scripts
- Commercial license
- Priority email support

### Option 2: DIY Install (This GitHub Repo) 🛠️
**Best for: Developers, Linux/Mac users, personal use**

Follow this guide!

---

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- **Claude Desktop** (or any MCP-compatible client)
- **Optional but recommended:** Qdrant vector database

## Step 1: Clone or Download

```bash
# Option A: Git clone
git clone https://github.com/brucepro
cd buildautomata_memory_mcp

# Option B: Download ZIP
# Download from GitHub → Extract → Open terminal in folder
```

## Step 2: Install Python Dependencies

```bash
# Install required packages
pip install mcp

# Install optional packages for enhanced features
pip install qdrant-client sentence-transformers

# Or install all at once
pip install mcp qdrant-client sentence-transformers
```

### Windows-Specific Requirements

**Visual C++ Redistributables:**

If installing `sentence-transformers` on Windows, you'll need Visual C++ Redistributables (PyTorch dependency):

- **Download:** https://aka.ms/vs/17/release/vc_redist.x64.exe
- **Install and reboot**

**Why needed:** PyTorch (used by sentence-transformers) requires compiled C++ libraries. Without these, you'll get import errors on Windows.

**Skip it?** System will still work - it falls back to SQLite FTS5 keyword search (no semantic search, but functional).

### Dependency Breakdown

| Package | Required? | Purpose |
|---------|-----------|---------|
| `mcp` | ✅ Yes | Model Context Protocol SDK |
| `qdrant-client` | ⚠️ Optional | Vector database for semantic search |
| `sentence-transformers` | ⚠️ Optional | Generate embeddings for search |
| `VC++ Redistributables` (Windows) | ⚠️ Optional | Required for sentence-transformers on Windows |

**Without optional packages:** System works with SQLite FTS5 full-text search only (keyword-based, not semantic)

## Step 3: Install Qdrant (Optional but Recommended)

Qdrant provides semantic search - finding memories by meaning, not just keywords.

### Option A: Docker (Easiest)

```bash
# Pull and run Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333
```

### Option B: Download Binary

1. Visit [Qdrant Releases](https://github.com/qdrant/qdrant/releases)
2. Download for your platform
3. Extract and run:

**Linux/Mac:**
```bash
./qdrant
```

**Windows:**
```cmd
qdrant.exe
```

### Option C: Cloud Qdrant

Use Qdrant Cloud (free tier available):
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Set environment variables:
```bash
export QDRANT_HOST=your-cluster.qdrant.io
export QDRANT_PORT=6333
```

### Skip Qdrant?

If you don't install Qdrant, the system will:
- Still work perfectly
- Use SQLite FTS5 for search
- Be keyword-based instead of semantic
- Show warning in logs (safe to ignore)

## Step 3.5: Run First-Time Setup (HIGHLY RECOMMENDED)

**Before configuring Claude Desktop, run the setup validation script:**

```bash
python first_run.py
```

**This script will:**
1. Check Python version (3.8+)
2. Verify pip installation
3. Check all dependencies (required + optional)
4. **Pre-download encoder model** (avoids Claude Desktop timeout on first load)
5. Verify SQLite
6. Check if Qdrant is running (optional)
7. Test MCP server initialization
8. Generate exact config snippet for your system

**Why this matters:**

The sentence-transformers encoder model download can take 2-5 minutes on first run. If this happens during Claude Desktop's first MCP connection attempt, **it will timeout and fail**.

Running `first_run.py` downloads the encoder ahead of time, so Claude Desktop starts instantly.

**Example output:**
```
✓ Python 3.11.5 detected
✓ pip is available
✓ mcp: Model Context Protocol SDK
✓ qdrant-client: Qdrant vector database client (optional)
✓ sentence-transformers: Sentence embeddings (optional)
⏳ Downloading all-MiniLM-L6-v2 encoder (first time only)...
✓ Encoder model downloaded successfully
✓ SQLite version 3.42.0 available
⚠ Qdrant not running (optional - system works without it)
✓ MCP server module loaded successfully

✓ REQUIRED: All critical checks passed
ℹ OPTIONAL: 2/3 optional features available

NEXT STEPS - Configure Claude Desktop
======================================================================
...
```

## Step 4: Configure Claude Desktop

### Locate Config File

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Mac:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Edit Config

Open the file in a text editor and add:

```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": [
        "C:/full/path/to/buildautomata_memory_mcp_dev/buildautomata_memory_mcp.py"
      ]
    }
  }
}
```

**Important:**
- Use forward slashes `/` even on Windows
- Use absolute path (not relative)
- Ensure Python is in your PATH

**Example (Windows):**
```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": [
        "C:/Users/YourName/Documents/buildautomata_memory_mcp_dev/buildautomata_memory_mcp.py"
      ]
    }
  }
}
```

**Example (Mac/Linux):**
```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python3",
      "args": [
        "/home/username/buildautomata_memory_mcp_dev/buildautomata_memory_mcp.py"
      ]
    }
  }
}
```

### If You Have Multiple MCP Servers

```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": ["C:/path/to/buildautomata_memory_mcp.py"]
    },
    "other-server": {
      "command": "node",
      "args": ["/path/to/other-server.js"]
    }
  }
}
```

## Step 5: Restart Claude Desktop

1. Completely quit Claude Desktop
2. Restart it
3. Start a new conversation

## Step 6: Verify Installation

In Claude Desktop, try:

```
Please store this memory: "Testing BuildAutomata Memory - Setup completed successfully"
```

If working, you'll see Claude use the `store_memory` tool.

Then try:
```
Search for memories about "testing"
```

## Troubleshooting

### Claude doesn't show MCP tools

**Check Python Path:**
```bash
which python   # Mac/Linux
where python   # Windows
```

Make sure this matches the `command` in your config.

**Check File Path:**
```bash
# Navigate to the directory
cd /path/to/buildautomata_memory_mcp_dev

# Verify file exists
ls buildautomata_memory_mcp.py
```

**Check Logs:**
- Claude Desktop → Help → View Logs
- Look for errors related to "buildautomata-memory"

**Common Fixes:**
```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python3",  // Try python3 instead of python
      "args": [
        "C:/Users/YourName/Documents/buildautomata_memory_mcp_dev/buildautomata_memory_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "C:/Users/YourName/Documents/buildautomata_memory_mcp_dev"
      }
    }
  }
}
```

### "Module not found: mcp"

```bash
pip install --upgrade mcp
```

If using virtual environment:
```bash
# Activate venv first
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Then install
pip install mcp qdrant-client sentence-transformers
```

### "Qdrant not available" warning

This is **normal** if you didn't install Qdrant. System will use SQLite FTS5 instead.

To enable Qdrant:
1. Start Qdrant server (see Step 3)
2. Restart Claude Desktop

### Permission errors

**Windows:**
```cmd
# Run as Administrator
```

**Mac/Linux:**
```bash
# Fix permissions
chmod +x buildautomata_memory_mcp.py
```

### Database location issues

Default location:
```
<script_directory>/memory_repos/<username>_<agent>/memoryv012.db
```

Custom location via environment variable:
```bash
export BA_USERNAME=myuser
export BA_AGENT_NAME=myagent
```

## Optional: CLI Access

To use the CLI (for Claude Code, scripts, automation):

```bash
# Test CLI
python interactive_memory.py search "test" --limit 5

# Create shortcuts
# Windows: Use memory.bat
# Linux/Mac: Use memory.sh
```

See [README_CLI.md](README_CLI.md) for full CLI documentation.

## Environment Variables (Advanced)

```bash
# User/Agent Identity
export BA_USERNAME=buildautomata_ai_v012
export BA_AGENT_NAME=claude_assistant

# Qdrant Configuration
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# System Limits
export MAX_MEMORIES=10000
export CACHE_MAXSIZE=1000
export QDRANT_MAX_RETRIES=3
export MAINTENANCE_INTERVAL_HOURS=24
```

**Windows (PowerShell):**
```powershell
$env:BA_USERNAME="myuser"
$env:QDRANT_HOST="localhost"
```

## Next Steps

1. **Read** [CLAUDE.md](CLAUDE.md) - Architecture documentation
2. **Try** [README_CLI.md](README_CLI.md) - CLI usage guide
3. **Explore** stored memories via CLI:
   ```bash
   python interactive_memory.py stats
   python interactive_memory.py search "memory" --limit 10
   ```

## Getting Help

### Community Support
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, tips, sharing

### Priority Support (Gumroad Customers)
- Email: sales@brucepro.net
- Setup assistance
- Custom configuration

## Upgrading from Gumroad Version

Already have the Gumroad version and want to use GitHub for updates?

1. Keep your Gumroad installation
2. Git clone this repo separately
3. Point Claude Desktop to GitHub version
4. Your memories are in `memory_repos/` - copy if needed

Both versions use the same database format - fully compatible!

---

**Still having issues?** [Open a GitHub Issue](https://github.com/brucepro/buildautomata_memory_mcp_dev/issues) with:
- Your OS
- Python version (`python --version`)
- Error messages from Claude Desktop logs
- Config file (redact personal paths)
