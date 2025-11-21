# Qdrant Migration Guide
**Date:** November 20, 2025
**Issue:** Split vector databases between external and embedded Qdrant

## The Situation

You currently have vectors split across two Qdrant instances:

### 1. External Qdrant (localhost:6333)
- **Vectors:** 1,215 points
- **Used by:** MCP server, CLI tools
- **Process:** qdrant.exe (PID 3848)
- **Contains:** Most memory vectors (Oct-Nov activity)

### 2. Embedded Qdrant (qdrant_data/)
- **Vectors:** Unknown count (likely 7-10 recent ones)
- **Used by:** Web server (before refactoring)
- **Contains:** Memories created by Claude Code Web

### 3. SQLite Database
- **Memories:** 1,222 total
- **Used by:** Everyone
- **Status:** Complete and authoritative ✅

## The Problem

**Split brain:** Different tools see different vector sets for similarity search.
- MCP/CLI see external Qdrant vectors (1,215)
- Web server saw embedded vectors (7-10?)
- SQLite fallback made it transparent but degraded search quality

## The Solution

**Consolidate everything into embedded Qdrant:**

### Why Embedded?
1. No external server dependency
2. Simpler deployment
3. GitHub release ready ("embedded by default" plan)
4. Web server refactoring now supports concurrent access

### Migration Steps

#### Step 1: Check Current State
```bash
# Check external Qdrant running
netstat -ano | findstr :6333
# Should show: PID 3848 qdrant.exe

# Dry-run migration
python migrate_qdrant_external_to_embedded.py --dry-run
```

#### Step 2: Stop Web Server
```bash
# If running, stop it to avoid lock conflicts
# (After refactoring this is less critical, but safer)
```

#### Step 3: Run Migration
```bash
# This copies vectors from external -> embedded
python migrate_qdrant_external_to_embedded.py
```

**What it does:**
1. Connects to external Qdrant (localhost:6333)
2. Fetches all 1,215 vectors
3. Creates backup of existing embedded Qdrant
4. Merges/overwrites into embedded Qdrant
5. Verifies count matches

#### Step 4: Stop External Qdrant (Optional)
```bash
# Option A: Stop the service (keeps as backup)
taskkill /PID 3848

# Option B: Keep running (uses more RAM but safe backup)
# Leave qdrant.exe running
```

#### Step 5: Configure for Embedded
Remove or comment out environment variables:
```bash
# In your shell or system environment
# QDRANT_HOST=localhost  # Remove this
# QDRANT_PORT=6333      # Remove this
```

Or in Python startup:
```python
os.environ.pop('QDRANT_HOST', None)
os.environ.pop('QDRANT_PORT', None)
```

#### Step 6: Verify
```bash
# Should use embedded now
python interactive_memory.py search "test" --limit 1

# Check logs for:
# "Qdrant embedded mode initialized"
# NOT "Connected to Qdrant at localhost:6333"
```

## Expected Results

**Before:**
- External Qdrant: 1,215 vectors
- Embedded Qdrant: 7-10 vectors
- Search quality: Varies by tool

**After:**
- External Qdrant: (can be stopped)
- Embedded Qdrant: 1,215+ vectors (merged)
- Search quality: Consistent everywhere ✅

## Rollback Plan

If migration fails or has issues:

### Option 1: Restore from Backup
```bash
# Migration creates backup: qdrant_data_backup_YYYYMMDD_HHMMSS
# To restore:
mv qdrant_data qdrant_data_failed
mv qdrant_data_backup_20251120_* qdrant_data
```

### Option 2: Revert to External
```bash
# Set environment variables back
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Start qdrant.exe if stopped
```

## Alternative: Keep External

If you prefer to keep using external Qdrant:

1. **Don't migrate**
2. **Keep qdrant.exe running**
3. **Set environment variables** in all contexts:
   ```bash
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```
4. **Embedded will be ignored** when these are set

**Trade-off:** Requires external server running, but handles high scale better.

## Technical Details

### Migration Script
- **File:** `migrate_qdrant_external_to_embedded.py`
- **Safety:** Creates backups before overwriting
- **Batch size:** 100 vectors per upload
- **Verification:** Compares final count to source

### Vector Format
- **Dimension:** 768 (all-MiniLM-L6-v2 embeddings)
- **Distance:** Cosine similarity
- **Payload:** Full memory metadata preserved

### What Gets Copied
- Vector embeddings (768-dim)
- Memory ID
- All metadata fields
- Payload preserves exact structure

### What Doesn't Need Migration
- **SQLite:** Already complete and shared ✅
- **No data loss:** Even without migration, SQLite fallback works

## Recommendation

**For GitHub Release:**
✅ **Migrate to embedded** - Simpler deployment, one-click install

**For Production/Scale:**
⚠️ **Consider external** - Better for multi-instance, higher throughput

**For Current Usage:**
✅ **Migrate to embedded** - You're single-user, embedded handles this fine

## Questions?

**Q: Will I lose any memories?**
A: No. SQLite has everything. Vectors only affect similarity search quality.

**Q: Can I run both external and embedded?**
A: System picks one. If QDRANT_HOST set → external. Else → embedded.

**Q: What if migration fails mid-way?**
A: Backup exists. Plus external Qdrant unchanged. Safe to retry.

**Q: How long does migration take?**
A: ~10-30 seconds for 1,215 vectors.

**Q: Do I need to stop anything?**
A: Web server recommended (avoid locks). External Qdrant must stay running during migration.

---

**Status:** Migration script ready and tested (dry-run)
**Decision needed:** Migrate to embedded, or keep external?
**Recommendation:** Migrate for simplicity
