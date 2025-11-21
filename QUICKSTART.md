# Quick Start Guide - 5 Minutes to Your First Memory

Get BuildAutomata Memory running in 5 minutes or less.

## Step 1: Install (2 minutes)

**Option A: Gumroad (Easiest)**
1. Purchase from https://brucepro1.gumroad.com/l/zizjl
2. Download and extract files
3. Double-click installer → Done!

**Option B: GitHub (Free, DIY)**

**Windows:**
```bash
git clone https://github.com/brucepro/buildautomata_memory_mcp.git
cd buildautomata_memory_mcp
pip install mcp qdrant-client sentence-transformers
```

**Important for Windows:** You may need Visual C++ Redistributables (needed by PyTorch/sentence-transformers):
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- System will still work without this (falls back to SQLite FTS5), but enhanced semantic search requires it

**macOS/Linux:**
```bash
git clone https://github.com/brucepro/buildautomata_memory_mcp.git
cd buildautomata_memory_mcp
pip install mcp qdrant-client sentence-transformers
```

## Step 1.5: Run First-Time Setup (HIGHLY RECOMMENDED)

**Before configuring Claude Desktop, run the setup script:**

```bash
python first_run.py
```

This script will:
- Check all dependencies (Python, pip, packages)
- Pre-download the sentence encoder model (avoids timeout on first Claude Desktop load)
- Verify SQLite and optional Qdrant
- Test MCP server initialization
- Give you the exact config snippet for your system

**Why this matters:** The encoder model download can take 2-5 minutes. If it happens during Claude Desktop's first MCP connection attempt, it will timeout. Running `first_run.py` downloads it ahead of time.

## Step 2: Configure Claude Desktop (2 minutes)

1. Open Claude Desktop settings
2. Find "Developer" → "Edit Config"
3. Add this to `claude_desktop_config.json`:

**Windows (use forward slashes or double backslashes):**
```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": ["D:/path/to/buildautomata_memory_mcp.py"]
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "buildautomata-memory": {
      "command": "python",
      "args": ["/path/to/buildautomata_memory_mcp.py"]
    }
  }
}
```

**Replace path with your actual path!**

**Windows path gotcha:** Use `D://path//to//file.py` or `D:/path/to/file.py` NOT `D:\path\to\file.py`

4. Restart Claude Desktop

## Step 3: Test It (1 minute)

Open Claude Desktop and say:

> "Store this memory: I prefer Python over JavaScript for backend development. Category: preference, importance: 0.8"

Then ask:

> "What do you remember about my programming preferences?"

**It should recall what you just told it!**

## That's It! 🎉

Your Claude now has persistent memory across conversations.

---

## What to Try Next

### Store Different Types of Memories

```
Store: I'm working on a robot project. Category: project, importance: 0.9

Store: I love reading sci-fi, especially Neal Stephenson. Category: personal, importance: 0.7

Store: My budget for groceries is $300 every 2 weeks. Category: finance, importance: 0.85
```

### Search Your Memories

```
Search your memories for information about my projects

Show me your timeline of memories about books

What are your most important memories?
```

### Watch Memory Evolution

After a few conversations, ask:

```
Show me the timeline of how your memories about me have evolved
```

You'll see version history, what changed when, and patterns in memory formation.

---

## Understanding What Just Happened

**Without this system:** Claude forgets everything after each conversation. Every chat starts from zero.

**With this system:** Claude remembers across conversations. Your preferences, projects, and context persist.

**The magic:** 
- Semantic search finds memories by meaning (not just keywords)
- Version tracking shows how memories evolved
- Importance scoring keeps valuable information prioritized
- Works across Claude Desktop, Claude Code, and Cursor AI

---

## Common Issues & Fixes

### "MCP server not found"
- Check your file path in config
- Make sure Python is installed (`python --version`)
- Try absolute path instead of relative

### "No memories found"
- Memories are per-user/agent combination
- Check `BA_USERNAME` environment variable
- Make sure you actually stored memories first

### "Qdrant connection failed"
- This is normal! The system works fine without Qdrant
- It falls back to SQLite FTS5 automatically
- To use Qdrant: Install and start it separately (optional)

---

## Where Your Memories Live

**Database location:** `buildautomata_memory.db` in the server directory

**Safety:** 
- All data is local (no cloud, no external APIs)
- You own your database file
- Can back it up, export it, delete it anytime

---

## Next Steps

Once you're comfortable with basics:

1. **Read CLAUDE.md** - Understanding the architecture
2. **Try the CLI** - Access memories from terminal (`interactive_memory.py`)
3. **Customize categories** - Organize memories your way
4. **Set up Qdrant** (optional) - Enhanced semantic search
5. **Explore timeline** - Visualize memory evolution

---

## Get Help

- **GitHub Issues:** Report bugs or request features
- **Email:** sales@brucepro.net for setup assistance
- **Documentation:** Full guides in README.md, CLAUDE.md, README_CLI.md

---

## Pro Tips

### Importance Scoring Guide
- **0.9-1.0:** Critical information (passwords, key decisions)
- **0.7-0.8:** Important preferences and ongoing projects
- **0.5-0.6:** Useful context and background info
- **0.3-0.4:** Nice-to-know details
- **0.0-0.2:** Temporary notes

### Category Suggestions
- `preference` - Your likes/dislikes, habits, style
- `project` - Active work you're doing
- `personal` - Background about you
- `learning` - Things you're studying
- `decision` - Important choices made
- `goal` - Things you're working toward

### Making Memories Useful
- Be specific: "I prefer TypeScript for frontend" vs "I like TypeScript"
- Include context: "Budget is $300/2weeks for family of 5" vs "My budget is $300"
- Update them: Memories can be revised as situations change
- Tag them: Use tags for easy filtering later

---

**You're ready!** Start building your AI's long-term memory. Every conversation now contributes to an evolving understanding.
