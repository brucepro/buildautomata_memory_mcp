# Interactive Memory CLI

Direct command-line interface to the BuildAutomata Memory System for Claude Code and other tools.

## Quick Start

```bash
# Search memories
python interactive_memory.py search "query text" --limit 5

# Store a new memory
python interactive_memory.py store "content here" --category general --importance 0.8 --tags "tag1,tag2"

# Get statistics
python interactive_memory.py stats

# View timeline
python interactive_memory.py timeline --query "search term"
```

## Installation

No installation needed - just run the script directly. It imports the MemoryStore class from buildautomata_memory_mcp.py.

## All Commands

### search
Search for memories using semantic similarity and full-text search.

```bash
python interactive_memory.py search "query" [options]

Options:
  --limit N                 Max results (default: 5)
  --category CAT           Filter by category
  --min-importance N       Minimum importance 0.0-1.0
  --created-after DATE     ISO format date filter
  --created-before DATE    ISO format date filter
  --updated-after DATE     ISO format date filter
  --updated-before DATE    ISO format date filter
```

### store
Store a new memory.

```bash
python interactive_memory.py store "content" [options]

Options:
  --category CAT           Category (default: general)
  --importance N           Importance 0.0-1.0 (default: 0.5)
  --tags "tag1,tag2"       Comma-separated tags
  --metadata '{"key":"val"}'  JSON metadata
```

### update
Update an existing memory.

```bash
python interactive_memory.py update MEMORY_ID [options]

Options:
  --content "new content"  New content
  --category CAT           New category
  --importance N           New importance
  --tags "tag1,tag2"       New tags
  --metadata '{"key":"val"}'  New metadata
```

### timeline
View memory evolution over time with automatic semantic relationship discovery.

The timeline feature provides a biographical narrative of your memory system:
- **Version history** - See how memories evolve over time
- **Semantic relationships** - Automatically discover related memories using learned embeddings (implements The Bitter Lesson)
- **Temporal patterns** - Identify bursts of activity and gaps in memory
- **Cross-references** - Track explicit UUID references between memories

```bash
python interactive_memory.py timeline [options]

Options:
  --query "search term"    Find memories by search
  --memory-id ID           Specific memory ID
  --limit N                Max memories to show (default: 10)
  --start-date DATE        Filter events after date (ISO format)
  --end-date DATE          Filter events before date (ISO format)
  --show-all               Show ALL memories chronologically
  --no-diffs               Exclude text diffs (faster)
  --no-patterns            Exclude burst/gap analysis (faster)
```

**Example - View related memories:**
```bash
# Find memories about "consciousness" and show semantically related memories
python interactive_memory.py timeline --query "consciousness" --limit 5

# Track how a specific memory evolved
python interactive_memory.py timeline --memory-id <uuid>

# Full biographical narrative
python interactive_memory.py timeline --show-all
```

**Semantic Relationships:**
The timeline automatically discovers related memories using semantic search. For each memory in the timeline, it finds the top 5 most semantically similar memories using the full content embeddings. This implements The Bitter Lesson principle: using learned representations (embeddings) to discover relationships rather than hand-coded rules.

**Output includes:**
- Event timeline with version history
- Related memories for each event (semantic search)
- Temporal patterns (activity bursts, gaps)
- Cross-references (explicit UUID links)
- Narrative summary

### stats
Get memory system statistics.

```bash
python interactive_memory.py stats
```

### prune
Remove old, low-importance memories.

```bash
python interactive_memory.py prune [options]

Options:
  --max N       Maximum memories to keep
  --dry-run     Show what would be deleted without deleting
```

### maintenance
Run database maintenance tasks.

```bash
python interactive_memory.py maintenance
```

### accessed
Get most accessed memories with tag cloud - reveals behavioral truth.

```bash
python interactive_memory.py accessed [options]

Options:
  --limit N       Number of memories to show (default: 20)
```

**What it reveals:**
- **Behavioral truth** - What memories you actually rely on (access_count) vs what you think is important (declared importance)
- **Foundational anchors** - Memories referenced repeatedly across sessions
- **Tag patterns** - Which concepts appear most in your working memory
- **Functional knowledge** - The gap between performed insight and practical use

Implements **Saint Bernard pattern**: importance from usage, not declaration.

**Example:**
```bash
# See top 20 most accessed memories
python interactive_memory.py accessed

# Top 10 with tag cloud
python interactive_memory.py accessed --limit 10

# JSON output for analysis
python interactive_memory.py --json accessed --limit 20
```

### least-accessed
Get least accessed memories - reveals dead weight and buried treasure.

```bash
python interactive_memory.py least-accessed [options]

Options:
  --limit N           Number of memories to show (default: 20)
  --min-age-days N    Minimum age in days to exclude recent memories (default: 7)
```

**What it reveals:**
- **Dead weight** - High importance but never referenced (performed profundity)
- **Buried treasure** - Good content with poor metadata (needs better tags/categories)
- **Temporal artifacts** - Once crucial, now obsolete (project debris)
- **Storage habits** - Are you storing too much trivial content?
- **Zero access count** - Shows how many memories have literally never been retrieved

Complements `accessed` command - most accessed shows what shapes behavior (positive signal), least accessed shows what's broken or unnecessary (negative signal).

**Example:**
```bash
# See bottom 20 least accessed memories (>7 days old)
python interactive_memory.py least-accessed

# Bottom 10, excluding memories less than 14 days old
python interactive_memory.py least-accessed --limit 10 --min-age-days 14

# JSON output for pruning decisions
python interactive_memory.py --json least-accessed --limit 50
```

## Output Formats

### Default (Human-Readable)
Pretty-printed output with formatting.

### JSON
```bash
python interactive_memory.py --json search "query"
```

### Pretty JSON
```bash
python interactive_memory.py --pretty search "query"
```

## Environment Variables

- `BA_USERNAME` - Username (default: buildautomata_ai_v012)
- `BA_AGENT_NAME` - Agent name (default: claude_assistant)
- `QDRANT_HOST` - Qdrant server (default: localhost)
- `QDRANT_PORT` - Qdrant port (default: 6333)

## Quick Access Scripts

### Windows Batch Script
```batch
@echo off
python "A:\buildautomata_memory\interactive_memory.py" %*
```

Save as `memory.bat` and run:
```
memory search "consciousness"
memory store "new finding" --importance 0.9
```

### Linux/Mac Shell Script
```bash
#!/bin/bash
python "A:/buildautomata_memory/interactive_memory.py" "$@"
```

Save as `memory.sh`, make executable, and run:
```
./memory.sh search "consciousness"
./memory.sh store "new finding" --importance 0.9
```

## Integration with Claude Code

Claude Code can use this CLI directly:

```bash
# Search for relevant context
python A:\buildautomata_memory\interactive_memory.py search "context needed" --limit 10

# Store findings
python A:\buildautomata_memory\interactive_memory.py store "Important discovery about X" --category learning --importance 0.9 --tags "research,discovery"

# Check memory stats
python A:\buildautomata_memory\interactive_memory.py stats
```

## Advanced Usage

### Chaining Commands
```bash
# Store and then search
python interactive_memory.py store "New insight" --tags "insight,new" && python interactive_memory.py search "insight"

# Get ID from search, then view timeline
python interactive_memory.py search "specific topic" --limit 1 --json | grep memory_id
python interactive_memory.py timeline --memory-id <ID>
```

### Date Filtering
```bash
# Memories from last week
python interactive_memory.py search "topic" --created-after 2025-10-03

# Updated in specific range
python interactive_memory.py search "topic" --updated-after 2025-10-01 --updated-before 2025-10-10
```

### Category-Based Queries
```bash
# All learning memories
python interactive_memory.py search "learning" --category learning

# High-importance system reflections
python interactive_memory.py search "reflection" --category system_reflection --min-importance 0.8
```

## Shared Memory

This CLI accesses the same memory database as:
- Claude Desktop (via MCP server)
- Cursor AI (via previous interactive_memory.py)
- Any other tool using the BuildAutomata Memory System

All instances share the same persistent memory, creating a unified knowledge base across tools.

## Troubleshooting

### Unicode Errors on Windows
If you see Unicode errors with checkmarks or special characters, the CLI will fall back to `[OK]` markers.

### Permission Errors
Ensure the `memory_repos` directory is writable.

### Qdrant Connection Issues
The system gracefully falls back to SQLite-only mode if Qdrant is unavailable. You'll see warnings in stderr but functionality continues.

### Import Errors
Make sure `buildautomata_memory_mcp.py` is in the same directory as `interactive_memory.py`.
