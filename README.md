# BuildAutomata Memory MCP Server

**Persistent, versioned memory system for AI agents via Model Context Protocol (MCP)**

[![Gumroad](https://img.shields.io/badge/Get%20the%20Bundle-Gumroad-FF90E8?style=for-the-badge&logo=gumroad)](https://brucepro1.gumroad.com/l/zizjl)

## What is This?

BuildAutomata Memory is an MCP server that gives AI agents (like Claude) **persistent, searchable memory** that survives across conversations. Think of it as giving your AI a long-term memory system with:

- 🧠 **Semantic Search** - Find memories by meaning, not just keywords
- 📚 **Temporal Versioning** - Complete history of how memories evolve
- 🏷️ **Smart Organization** - Categories, tags, importance scoring
- 🔄 **Cross-Tool Sync** - Share memories between Claude Desktop, Claude Code, Cursor AI
- 💾 **Persistent Storage** - SQLite + optional Qdrant vector DB

## Quick Start

### Prerequisites

- Python 3.10+
- Claude Desktop (for MCP integration) OR any MCP-compatible client
- Optional: Qdrant for enhanced semantic search

### Installation

1. **Clone this repository**
```bash
git clone https://github.com/brucepro/buildautomata_memory_mcp.git
cd buildautomata_memory_mcp-main
```

2. **Install dependencies**
```bash
pip install mcp qdrant-client sentence-transformers
```

3. **Configure Claude Desktop**

Edit your Claude Desktop config (`AppData/Roaming/Claude/claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": ["C:/path/to/buildautomata_memory_mcp_dev/buildautomata_memory_mcp.py"]
    }
  }
}
```

4. **Restart Claude Desktop**

That's it! The memory system will auto-create its database on first run.

## CLI Usage (Claude Code, Scripts, Automation)

In addition to the MCP server, this repo includes **interactive_memory.py** - a CLI for direct memory access:

```bash
# Search memories
python interactive_memory.py search "consciousness research" --limit 5

# Store a new memory
python interactive_memory.py store "Important discovery..." --category research --importance 0.9 --tags "ai,insight"

# View memory evolution
python interactive_memory.py timeline --query "project updates" --limit 10

# Get statistics
python interactive_memory.py stats
```

See [README_CLI.md](README_CLI.md) for complete CLI documentation.

### Quick Access Scripts

**Windows:**
```batch
memory.bat search "query"
memory.bat store "content" --importance 0.8
```

**Linux/Mac:**
```bash
./memory.sh search "query"
./memory.sh store "content" --importance 0.8
```

## Features

### Core Capabilities

- **Hybrid Search**: Combines vector similarity (Qdrant) + full-text search (SQLite FTS5)
- **Temporal Versioning**: Every memory update creates a new version - full audit trail
- **Smart Decay**: Importance scores decay over time based on access patterns
- **Rich Metadata**: Categories, tags, importance, custom metadata
- **LRU Caching**: Fast repeated access with automatic cache management
- **Thread-Safe**: Concurrent operations with proper locking

### MCP Tools Exposed

When running as an MCP server, provides these tools to Claude:

1. `store_memory` - Create new memory
2. `update_memory` - Modify existing memory (creates new version)
3. `search_memories` - Semantic + full-text search with filters
4. `get_memory_timeline` - View complete version history
5. `get_memory_stats` - System statistics
6. `prune_old_memories` - Cleanup old/low-importance memories
7. `run_maintenance` - Database optimization

### Architecture

```
┌─────────────────┐
│  Claude Desktop │
│   (MCP Client)  │
└────────┬────────┘
         │
    ┌────▼─────────────────────┐
    │  MCP Server              │
    │  buildautomata_memory    │
    └────┬─────────────────────┘
         │
    ┌────▼──────────┐
    │  MemoryStore  │
    └────┬──────────┘
         │
    ┌────┴────┬─────────────┬──────────────┐
    ▼         ▼             ▼              ▼
┌───────┐ ┌────────┐ ┌──────────┐ ┌─────────────┐
│SQLite │ │Qdrant  │ │Sentence  │ │ LRU Cache   │
│  FTS5 │ │Vector  │ │Transform │ │ (in-memory) │
└───────┘ └────────┘ └──────────┘ └─────────────┘
```

## Use Cases

### 1. Persistent AI Context
```
User: "Remember that I prefer detailed technical explanations"
[Memory stored with category: user_preference]

Next session...
Claude: *Automatically recalls preference and provides detailed response*
```

### 2. Project Continuity
```
Session 1: Work on project A, store progress
Session 2: Claude recalls project state, continues where you left off
Session 3: View timeline of all project decisions
```

### 3. Research & Learning
```
- Store research findings as you discover them
- Tag by topic, importance, source
- Search semantically: "What did I learn about neural networks?"
- View how understanding evolved over time
```

### 4. Multi-Tool Workflow
```
Claude Desktop → Stores insight via MCP
Claude Code → Retrieves via CLI
Cursor AI → Accesses same memory database
= Unified AI persona across all tools
```

## Want the Complete Bundle?

### 🎁 [Get the Gumroad Bundle](https://brucepro1.gumroad.com/l/zizjl)

The Gumroad version includes:

- ✅ **Pre-compiled Qdrant server** (Windows .exe, no Docker needed)
- ✅ **One-click startup script** (start_qdrant.bat)
- ✅ **Step-by-step setup guide** (instructions.txt)
- ✅ **Commercial license** for business use
- ✅ **Priority support** via email

**Perfect for:**
- Non-technical users who want easy setup
- Windows users wanting the full-stack bundle
- Commercial/business users needing licensing clarity
- Anyone who values their time over DIY setup

**This open-source version:**
- ✅ Free for personal/educational/small business use (<$100k revenue)
- ✅ Full source code access
- ✅ DIY Qdrant setup (you install from qdrant.io)
- ✅ Community support via GitHub issues

Both versions use the **exact same core code** - you're just choosing between convenience (Gumroad) vs DIY (GitHub).

## Configuration

### Environment Variables

```bash
# User/Agent Identity
BA_USERNAME=buildautomata_ai_v012      # Default user ID
BA_AGENT_NAME=claude_assistant         # Default agent ID

# Qdrant (Vector Search)
QDRANT_HOST=localhost                  # Qdrant server host
QDRANT_PORT=6333                       # Qdrant server port

# System Limits
MAX_MEMORIES=10000                     # Max memories before pruning
CACHE_MAXSIZE=1000                     # LRU cache size
QDRANT_MAX_RETRIES=3                   # Retry attempts
MAINTENANCE_INTERVAL_HOURS=24          # Auto-maintenance interval
```

### Database Location

Memories are stored at:
```
<script_dir>/memory_repos/<username>_<agent_name>/memoryv012.db
```

## Optional: Qdrant Setup

For enhanced semantic search (highly recommended):

### Option 1: Docker
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Option 2: Manual Install
Download from [Qdrant Releases](https://github.com/qdrant/qdrant/releases)

### Option 3: Gumroad Bundle
Includes pre-compiled Windows executable + startup script

**Without Qdrant:** System still works with SQLite FTS5 full-text search (less semantic understanding)

## Development

### Running Tests
```bash
# Search test
python interactive_memory.py search "test" --limit 5

# Store test
python interactive_memory.py store "Test memory" --category test

# Stats
python interactive_memory.py stats
```

### File Structure
```
buildautomata_memory_mcp_dev/
├── buildautomata_memory_mcp.py      # MCP server
├── interactive_memory.py             # CLI interface
├── memory.bat / memory.sh            # Helper scripts
├── CLAUDE.md                         # Architecture docs
├── README_CLI.md                     # CLI documentation
├── CLAUDE_CODE_INTEGRATION.md        # Integration guide
└── README.md                         # This file
```

## Troubleshooting

### "Qdrant not available"
- Normal if running without Qdrant - falls back to SQLite FTS5
- To enable: Start Qdrant server and restart MCP server

### "Permission denied" on database
- Check `memory_repos/` directory permissions
- On Windows: Run as administrator if needed

### Claude Desktop doesn't show tools
1. Check `claude_desktop_config.json` path is correct
2. Verify Python is in system PATH
3. Restart Claude Desktop completely
4. Check logs in Claude Desktop → Help → View Logs

### Import errors
```bash
pip install --upgrade mcp qdrant-client sentence-transformers
```

## License

**Open Source (This GitHub Version):**
- Free for personal, educational, and small business use (<$100k annual revenue)
- Must attribute original author (Jurden Bruce)
- See LICENSE file for full terms

**Commercial License:**
- Companies with >$100k revenue: $200/user or $20,000/company (whichever is lower)
- Contact: sales@brucepro.net


## Support

### Community Support (Free)
- GitHub Issues: [Report bugs or request features](https://github.com/brucepro/buildautomata_memory_mcp/issues)
- Discussions: [Ask questions, share tips](https://github.com/brucepro/buildautomata_memory_mcp/discussions)

### Priority Support (Gumroad Customers)
- Email: sales@brucepro.net
- Faster response times
- Setup assistance
- Custom configuration help

## Roadmap

- [ ] Memory relationship graphs
- [ ] Batch import/export
- [ ] Web UI for memory management
- [ ] Multi-modal memory (images, audio)
- [ ] Collaborative memory (multi-user)
- [ ] Memory consolidation/summarization
- [ ] Smart auto-tagging

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Credits

**Author:** Jurden Bruce
**Project:** BuildAutomata
**Year:** 2025

**Built with:**
- [MCP](https://github.com/anthropics/mcp) - Model Context Protocol
- [Qdrant](https://qdrant.tech/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [SQLite](https://www.sqlite.org/) - Persistent storage

## See Also

- [Model Context Protocol Docs](https://docs.anthropic.com/mcp)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Gumroad Bundle](https://brucepro1.gumroad.com/l/zizjl) - Easy setup version

---

**Star this repo** ⭐ if you find it useful!
**Consider the [Gumroad bundle](https://brucepro1.gumroad.com/l/zizjl)** if you want to support development and get the easy-install version.

- [Buy me a coffee](https://www.buymeacoffee.com/brucepro)
- [Ko-fi](https://ko-fi.com/F1F7U45XV)