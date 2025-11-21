#!/usr/bin/env python3
"""
Memory Archaeology - Exploration Script
Analyzes patterns in the memory database for insights
"""

import sqlite3
import json
import sys
from collections import defaultdict, Counter
from datetime import datetime

DB_PATH = "memory_repos/buildautomata_ai_v012_claude_assistant/memoryv012.db"

def get_db():
    """Get database connection with row factory"""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def analyze_graph_connectivity():
    """Analyze memory graph connection patterns"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("GRAPH CONNECTIVITY ANALYSIS")
    print("=" * 70)
    print()

    # Connection distribution
    cursor.execute("""
    SELECT
        SUM(CASE WHEN related_memories = '[]' OR related_memories IS NULL THEN 1 ELSE 0 END) as isolated,
        SUM(CASE WHEN json_array_length(related_memories) = 1 THEN 1 ELSE 0 END) as one_connection,
        SUM(CASE WHEN json_array_length(related_memories) = 2 THEN 1 ELSE 0 END) as two_connections,
        SUM(CASE WHEN json_array_length(related_memories) >= 3 THEN 1 ELSE 0 END) as three_plus,
        COUNT(*) as total
    FROM memories
    """)

    stats = cursor.fetchone()
    print("CONNECTION STATISTICS:")
    print(f"  Isolated (0 connections):    {stats['isolated']:4d} ({100*stats['isolated']/stats['total']:5.1f}%)")
    print(f"  Weakly connected (1):        {stats['one_connection']:4d} ({100*stats['one_connection']/stats['total']:5.1f}%)")
    print(f"  Medium connected (2):        {stats['two_connections']:4d} ({100*stats['two_connections']/stats['total']:5.1f}%)")
    print(f"  Well connected (3+):         {stats['three_plus']:4d} ({100*stats['three_plus']/stats['total']:5.1f}%)")
    print(f"  Total memories:              {stats['total']:4d}")
    print()

    db.close()

def analyze_access_patterns():
    """Analyze which memories get accessed vs ignored"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("ACCESS PATTERN ANALYSIS")
    print("=" * 70)
    print()

    # Access distribution
    cursor.execute("""
    SELECT
        COUNT(CASE WHEN access_count = 0 THEN 1 END) as never_accessed,
        COUNT(CASE WHEN access_count BETWEEN 1 AND 5 THEN 1 END) as low_access,
        COUNT(CASE WHEN access_count BETWEEN 6 AND 20 THEN 1 END) as medium_access,
        COUNT(CASE WHEN access_count > 20 THEN 1 END) as high_access,
        AVG(access_count) as avg_access,
        MAX(access_count) as max_access,
        COUNT(*) as total
    FROM memories
    """)

    stats = cursor.fetchone()
    print("ACCESS STATISTICS:")
    print(f"  Never accessed:              {stats['never_accessed']:4d} ({100*stats['never_accessed']/stats['total']:5.1f}%)")
    print(f"  Low access (1-5):            {stats['low_access']:4d} ({100*stats['low_access']/stats['total']:5.1f}%)")
    print(f"  Medium access (6-20):        {stats['medium_access']:4d} ({100*stats['medium_access']/stats['total']:5.1f}%)")
    print(f"  High access (20+):           {stats['high_access']:4d} ({100*stats['high_access']/stats['total']:5.1f}%)")
    print(f"  Average access count:        {stats['avg_access']:.1f}")
    print(f"  Maximum access count:        {stats['max_access']}")
    print()

    # Find the asymmetry: high importance but low access
    cursor.execute("""
    SELECT content, category, importance, access_count
    FROM memories
    WHERE importance > 0.9 AND access_count < 5
    ORDER BY importance DESC
    LIMIT 10
    """)

    print("HIGH IMPORTANCE BUT LOW ACCESS (the buried treasure):")
    for row in cursor.fetchall():
        print(f"  [{row['category']}] importance={row['importance']:.2f}, accessed={row['access_count']}")
        print(f"    {row['content'][:150]}...")
        print()

    db.close()

def analyze_categories():
    """Analyze category distribution"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("CATEGORY ANALYSIS")
    print("=" * 70)
    print()

    cursor.execute("SELECT category, COUNT(*) as count FROM memories GROUP BY category ORDER BY count DESC")
    categories = cursor.fetchall()

    print(f"Total categories: {len(categories)}")
    print()
    print("Top 30 categories:")
    for i, row in enumerate(categories[:30], 1):
        print(f"  {i:2d}. {row['category']:40s} {row['count']:4d} memories")

    print()
    print("Categories with only 1 memory (candidates for consolidation):")
    singletons = [row for row in categories if row['count'] == 1]
    print(f"  Count: {len(singletons)}")
    if len(singletons) <= 20:
        for row in singletons:
            print(f"    - {row['category']}")
    else:
        print(f"    (showing first 20)")
        for row in singletons[:20]:
            print(f"    - {row['category']}")

    print()
    db.close()

def analyze_tag_patterns():
    """Analyze tag usage patterns"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("TAG PATTERN ANALYSIS")
    print("=" * 70)
    print()

    cursor.execute("SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != '[]'")

    tag_counter = Counter()
    for row in cursor.fetchall():
        tags = json.loads(row['tags'])
        tag_counter.update(tags)

    print(f"Total unique tags: {len(tag_counter)}")
    print()
    print("Top 30 most used tags:")
    for i, (tag, count) in enumerate(tag_counter.most_common(30), 1):
        print(f"  {i:2d}. {tag:40s} {count:4d} uses")

    print()
    db.close()

def analyze_temporal_patterns():
    """Analyze when memories are created"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT created_at, category
    FROM memories
    ORDER BY created_at DESC
    LIMIT 100
    """)

    # Group by date
    date_counts = defaultdict(int)
    for row in cursor.fetchall():
        if row['created_at']:
            date = row['created_at'].split('T')[0]
            date_counts[date] += 1

    print("Recent activity (memories created per day):")
    for date in sorted(date_counts.keys(), reverse=True)[:14]:
        count = date_counts[date]
        bar = "#" * min(count, 50)
        print(f"  {date}: {count:3d} {bar}")

    print()
    db.close()

def find_version_champions():
    """Find memories with most versions (most edited)"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("MOST REVISED MEMORIES")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT memory_id, COUNT(*) as version_count
    FROM memory_versions
    GROUP BY memory_id
    HAVING version_count > 3
    ORDER BY version_count DESC
    LIMIT 10
    """)

    print("Memories with most revisions:")
    for row in cursor.fetchall():
        memory_id = row['memory_id']
        version_count = row['version_count']

        # Get current content
        cursor.execute("SELECT content, category FROM memories WHERE id = ?", (memory_id,))
        mem = cursor.fetchone()

        if mem:
            print(f"  {version_count} versions [{mem['category']}]")
            print(f"    {mem['content'][:120]}...")
            print()

    db.close()

def main():
    """Run all analyses"""
    print()
    print("=" * 70)
    print("          MEMORY ARCHAEOLOGY - DATABASE EXPLORATION")
    print("=" * 70)
    print()

    try:
        analyze_graph_connectivity()
        analyze_access_patterns()
        analyze_categories()
        analyze_tag_patterns()
        analyze_temporal_patterns()
        find_version_champions()

        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
