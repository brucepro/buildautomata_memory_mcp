# BuildAutomata Memory System - Backup & Restore

Complete backup and restore system for the BuildAutomata Memory MCP server.

---

## Quick Start

### Create a Backup

```bash
python backup_memory.py --description "Before major changes"
```

### List Backups

```bash
python backup_memory.py --list
```

### Restore from Backup

```bash
# Dry run first (safe, no changes)
python restore_memory.py backup_20251024_191534.zip --dry-run

# Actual restore (DESTRUCTIVE - deletes all data!)
python restore_memory.py backup_20251024_191534.zip
```

---

## Features

### Backup (`backup_memory.py`)

**What it backs up:**
- SQLite database (`memoryv012.db`) - All memories, versions, intentions
- Qdrant collection - All vector embeddings and payloads
- Manifest - Backup metadata, timestamps, statistics

**Output:**
- Creates timestamped zip file in `backups/` directory
- Format: `backup_YYYYMMDD_HHMMSS.zip`
- Includes full statistics and description

**Options:**
```bash
python backup_memory.py [OPTIONS]

  --description, -d TEXT    Backup description
  --list, -l               List available backups
  --username TEXT          Override BA_USERNAME
  --agent TEXT             Override BA_AGENT_NAME
```

**Example:**
```bash
# Create backup with description
python backup_memory.py --description "Pre-production snapshot"

# Backup for specific user/agent
python backup_memory.py --username myuser --agent myagent
```

---

### Restore (`restore_memory.py`)

**What it does:**
1. Validates backup archive
2. **DELETES** all existing data (with confirmation)
3. Restores SQLite database
4. Recreates Qdrant collection
5. Imports all vectors

**Safety Features:**
- Dry-run mode (test without changes)
- Requires explicit confirmation: `DELETE ALL DATA`
- Validates backup before proceeding
- Reports detailed statistics

**Options:**
```bash
python restore_memory.py BACKUP [OPTIONS]

  BACKUP                   Backup file to restore
  --dry-run               Validate without restoring
  --list, -l              List available backups
  --username TEXT          Override BA_USERNAME
  --agent TEXT             Override BA_AGENT_NAME
```

**Example:**
```bash
# Safe dry-run first
python restore_memory.py backup_20251024_191534.zip --dry-run

# Actual restore (you'll be asked to confirm)
python restore_memory.py backup_20251024_191534.zip
```

---

## Qdrant Mode Detection

The backup and restore tools automatically detect your Qdrant configuration:

### Embedded Mode (Default)
- **No configuration needed** - Works out of the box
- Qdrant data stored locally in `memory_repos/{user}_{agent}/qdrant_data/`
- No external server required

### External Mode
Set environment variable to use external Qdrant server:
```bash
export USE_EXTERNAL_QDRANT=true
export QDRANT_HOST=localhost     # Optional, defaults to localhost
export QDRANT_PORT=6333          # Optional, defaults to 6333
```

**Note:** Both backup and restore will use the same mode as your current configuration. To backup from external and restore to embedded (or vice versa), change the environment variables before running the restore command.

---

## Use Cases

### 1. Regular Backups

**Recommended:** Create backups before major changes

```bash
# Before implementing new features
python backup_memory.py --description "Before agency bridge refactor"

# Before bulk data operations
python backup_memory.py --description "Before memory consolidation"

# Weekly snapshots
python backup_memory.py --description "Weekly backup - Week 43"
```

### 2. Version Snapshots

**Save milestones:**

```bash
python backup_memory.py --description "Production release v1.0"
python backup_memory.py --description "After 1000 memories milestone"
python backup_memory.py --description "End of October 2025 session"
```

### 3. Migration

**Move between systems:**

```bash
# On old system
python backup_memory.py --description "Migration export"

# Copy backup_*.zip to new system

# On new system
python restore_memory.py backup_20251024_191534.zip
```

### 4. Disaster Recovery

**Restore to known-good state:**

```bash
# Something went wrong, restore yesterday's backup
python backup_memory.py --list  # Find the backup
python restore_memory.py backup_20251023_180000.zip
```

### 5. Testing & Experimentation

**Safe experimentation:**

```bash
# Save current state
python backup_memory.py --description "Before experiment"

# Run experiments, make changes...

# Restore if needed
python restore_memory.py backup_20251024_120000.zip
```

---

## Backup Contents

Each backup is a ZIP file containing:

### 1. `memoryv012.db` (SQLite)
- Complete database with all tables
- All memories (current + historical versions)
- All intentions
- Full-text search indexes
- Timestamps and metadata

### 2. `qdrant_vectors.json` (Qdrant)
- All vector embeddings
- Complete payloads (memory content, metadata)
- Collection configuration (vector size, distance metric)

### 3. `manifest.json` (Metadata)
```json
{
  "timestamp": "20251024_191534",
  "created_at": "2025-10-24T19:15:51.845895",
  "description": "Backup description here",
  "config": {
    "username": "buildautomata_ai_v012",
    "agent_name": "claude_assistant",
    "collection_name": "buildautomata_ai_v012_claude_assistant_memories"
  },
  "stats": {
    "sqlite": {
      "total_memories": 410,
      "total_versions": 429,
      "total_intentions": 5
    },
    "qdrant": {
      "total_vectors": 420
    }
  }
}
```

---

## Technical Details

### Backup Process

1. **SQLite Export:**
   - File copy of database (atomic, fast)
   - Reads statistics for manifest
   - Preserves all schema, indexes, data

2. **Qdrant Export:**
   - Uses `scroll()` API to iterate all points
   - Exports in batches of 100
   - Saves vectors + payloads to JSON
   - Includes collection configuration

3. **Archive Creation:**
   - Creates ZIP with all files
   - Uses compression (ZIP_DEFLATED)
   - Generates manifest with checksums
   - Timestamped filename

### Restore Process

1. **Validation:**
   - Checks ZIP file integrity
   - Verifies manifest exists
   - Validates required files present

2. **Confirmation:**
   - Shows what will be deleted
   - Requires user to type `DELETE ALL DATA`
   - Can be skipped with `--dry-run`

3. **Data Deletion:**
   - Removes SQLite database file
   - Deletes Qdrant collection (`delete_collection()`)
   - Ensures clean slate

4. **SQLite Restore:**
   - Extracts database from ZIP
   - Places in correct directory
   - Verifies record counts

5. **Qdrant Restore:**
   - Creates new collection (same config)
   - Imports vectors in batches
   - Uses `upsert()` for reliability
   - Verifies final count

---

## Configuration

### Environment Variables

Same as main memory system:

- `BA_USERNAME` - Default: `buildautomata_ai_v012`
- `BA_AGENT_NAME` - Default: `claude_assistant`
- `QDRANT_HOST` - Default: `localhost`
- `QDRANT_PORT` - Default: `6333`

### Directory Structure

```
buildautomata_memory/
├── backup_memory.py          # Backup script
├── restore_memory.py         # Restore script
├── backups/                  # Backup archives (created)
│   ├── backup_20251024_191534.zip
│   ├── backup_20251024_120000.zip
│   └── ...
└── memory_repos/             # Live database
    └── buildautomata_ai_v012_claude_assistant/
        └── memoryv012.db
```

---

## Safety & Best Practices

### ✓ DO

- Create backups before major changes
- Use descriptive backup descriptions
- Test restore with `--dry-run` first
- Keep multiple backup versions
- Store backups in multiple locations (external drive, cloud)
- Verify backup success (check size, stats)

### ✗ DON'T

- Don't rely on a single backup
- Don't skip the dry-run before restore
- Don't ignore confirmation prompts
- Don't delete backups immediately after restore
- Don't restore without checking what's in the backup first

### Backup Retention

**Suggested strategy:**

- Keep daily backups for 7 days
- Keep weekly backups for 1 month
- Keep monthly backups for 1 year
- Keep milestone backups indefinitely

**Manual cleanup:**
```bash
# List backups with details
python backup_memory.py --list

# Delete old backups manually from backups/ directory
rm backups/backup_20251001_*.zip
```

---

## Troubleshooting

### Backup Fails

**"Database not found":**
- Check `BA_USERNAME` and `BA_AGENT_NAME` are correct
- Verify database exists in `memory_repos/` directory

**"Qdrant export failed":**
- Check Qdrant server is running
- Verify `QDRANT_HOST` and `QDRANT_PORT` are correct
- Backup will still complete with SQLite only

### Restore Fails

**"Backup not found":**
- Use full filename: `backup_20251024_191534.zip`
- Or provide full path: `/path/to/backup.zip`

**"Collection creation failed":**
- Ensure Qdrant server is running
- Check collection doesn't already exist
- Restore will complete SQLite, skip Qdrant

**"Confirmation cancelled":**
- This is safety feature - type exactly: `DELETE ALL DATA`

---

## Examples

### Complete Backup/Restore Cycle

```bash
# 1. Create backup
python backup_memory.py --description "Before cleanup"

# Output:
# [SUCCESS] Backup created successfully!
#   Location: backups/backup_20251024_191534.zip
#   Size: 2.99 MB
#   Memories: 410
#   Versions: 429
#   Vectors: 420

# 2. Make changes, experiment...

# 3. Something went wrong, restore
python restore_memory.py backup_20251024_191534.zip --dry-run

# Output:
# [DRY RUN] Would restore:
#   SQLite: 410 memories
#   Qdrant: 420 vectors

# 4. Looks good, do actual restore
python restore_memory.py backup_20251024_191534.zip

# Prompt:
# Type 'DELETE ALL DATA' to confirm: DELETE ALL DATA

# Output:
# [SUCCESS] Restore completed successfully!
#   Restored from: backup_20251024_191534.zip
#   Memories: 410
#   Versions: 429
#   Vectors: 420
```

---

## Integration

### Automated Backups

**Cron job (Linux/Mac):**
```bash
# Daily backup at 2 AM
0 2 * * * cd /path/to/buildautomata_memory && python backup_memory.py --description "Daily automated backup"
```

**Task Scheduler (Windows):**
```batch
cd A:\buildautomata_memory
python backup_memory.py --description "Daily automated backup"
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
python backup_memory.py --description "Pre-commit snapshot"
```

---

## Version History

**v1.0 - October 24, 2025**
- Initial release
- SQLite + Qdrant backup/restore
- Dry-run support
- Confirmation prompts
- Manifest with statistics
- Batch processing for vectors

---

**For questions or issues, see main project documentation in `CLAUDE.md`**
