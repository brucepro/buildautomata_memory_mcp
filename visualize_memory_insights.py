#!/usr/bin/env python3
"""
Memory Insights Visualization
Creates ASCII charts and insights about memory patterns
"""

import sqlite3
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta

DB_PATH = "memory_repos/buildautomata_ai_v012_claude_assistant/memoryv012.db"

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def visualize_importance_vs_access():
    """Show the Saint Bernard pattern: high importance but low access"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("IMPORTANCE VS ACCESS PATTERN (The Saint Bernard Effect)")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT id, content, category, importance, access_count
    FROM memories
    WHERE importance > 0.9
    ORDER BY access_count ASC, importance DESC
    LIMIT 15
    """)

    print("High importance memories ranked by how rarely they're accessed:")
    print("(These are the buried treasures)\n")

    for i, row in enumerate(cursor.fetchall(), 1):
        importance_bar = "#" * int(row['importance'] * 20)
        access_bar = "." * min(row['access_count'], 40)

        print(f"{i:2d}. [{row['category'][:30]:30s}]")
        print(f"    Importance: {row['importance']:.2f} {importance_bar}")
        print(f"    Accessed:   {row['access_count']:3d} times {access_bar}")
        print(f"    Content: {row['content'][:80]}...")
        print()

    db.close()

def visualize_access_distribution():
    """Show distribution of access counts"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("ACCESS COUNT DISTRIBUTION")
    print("=" * 70)
    print()

    cursor.execute("SELECT access_count FROM memories")
    access_counts = [row['access_count'] for row in cursor.fetchall()]

    # Create buckets
    buckets = [
        (0, 0, "Never"),
        (1, 5, "Rare (1-5)"),
        (6, 20, "Occasional (6-20)"),
        (21, 50, "Frequent (21-50)"),
        (51, 100, "Very Frequent (51-100)"),
        (101, 999, "Obsessive (100+)"),
    ]

    print("How many times have memories been accessed?\n")

    for min_val, max_val, label in buckets:
        count = sum(1 for c in access_counts if min_val <= c <= max_val)
        pct = 100 * count / len(access_counts)
        bar = "#" * int(pct)

        print(f"{label:25s} {count:4d} ({pct:5.1f}%) {bar}")

    print()
    db.close()

def visualize_category_health():
    """Show category usage patterns"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("CATEGORY HEALTH CHECK")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT category, COUNT(*) as count, AVG(importance) as avg_importance, AVG(access_count) as avg_access
    FROM memories
    GROUP BY category
    ORDER BY count DESC
    LIMIT 20
    """)

    print("Top 20 categories by memory count:\n")
    print(f"{'Category':<35s} {'Count':>6s} {'Avg Imp':>8s} {'Avg Access':>10s}")
    print("-" * 70)

    for row in cursor.fetchall():
        count_bar = "#" * min(row['count'], 25)
        print(f"{row['category']:<35s} {row['count']:6d} {row['avg_importance']:8.2f} {row['avg_access']:10.1f} {count_bar}")

    print()

    # Singleton categories
    cursor.execute("""
    SELECT COUNT(*) as singleton_count
    FROM (SELECT category, COUNT(*) as cnt FROM memories GROUP BY category HAVING cnt = 1)
    """)

    singleton_count = cursor.fetchone()['singleton_count']
    print(f"Singleton categories (only 1 memory): {singleton_count}")
    print(f"Category fragmentation score: {singleton_count / 438 * 100:.1f}% are singletons")
    print()

    db.close()

def visualize_temporal_bursts():
    """Show when memories were created - find bursts and gaps"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("TEMPORAL ACTIVITY PATTERNS")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT created_at, category
    FROM memories
    ORDER BY created_at DESC
    LIMIT 200
    """)

    # Group by date
    date_counts = defaultdict(int)
    for row in cursor.fetchall():
        if row['created_at']:
            date = row['created_at'].split('T')[0]
            date_counts[date] += 1

    print("Recent activity (last 30 days):\n")
    print(f"{'Date':<12s} {'Memories':>9s} {'Activity Graph'}")
    print("-" * 70)

    for date in sorted(date_counts.keys(), reverse=True)[:30]:
        count = date_counts[date]
        bar = "#" * min(count, 50)
        print(f"{date:<12s} {count:9d} {bar}")

    print()

    # Find gaps (days with no activity)
    all_dates = sorted(date_counts.keys())
    if len(all_dates) > 1:
        first_date = datetime.fromisoformat(all_dates[0])
        last_date = datetime.fromisoformat(all_dates[-1])
        total_days = (last_date - first_date).days

        print(f"Activity span: {all_dates[0]} to {all_dates[-1]} ({total_days} days)")
        print(f"Active days: {len(date_counts)}")
        print(f"Silent days: {total_days - len(date_counts)}")
        print()

    db.close()

def visualize_tag_insights():
    """Show tag usage patterns"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("TAG INSIGHTS")
    print("=" * 70)
    print()

    cursor.execute("SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != '[]'")

    tag_counter = Counter()
    for row in cursor.fetchall():
        tags = json.loads(row['tags'])
        tag_counter.update(tags)

    print(f"Total unique tags: {len(tag_counter)}")
    print(f"Total memories with tags: {sum(tag_counter.values())}")
    print()

    print("Top 20 tags:\n")
    print(f"{'Tag':<40s} {'Uses':>6s}")
    print("-" * 70)

    for tag, count in tag_counter.most_common(20):
        bar = "#" * min(count // 5, 30)
        print(f"{tag:<40s} {count:6d} {bar}")

    print()
    db.close()

def visualize_graph_structure():
    """Show memory graph connectivity"""
    db = get_db()
    cursor = db.cursor()

    print("=" * 70)
    print("MEMORY GRAPH STRUCTURE")
    print("=" * 70)
    print()

    cursor.execute("""
    SELECT
        CASE
            WHEN related_memories = '[]' OR related_memories IS NULL THEN 0
            ELSE json_array_length(related_memories)
        END as connection_count,
        COUNT(*) as memory_count
    FROM memories
    GROUP BY connection_count
    ORDER BY connection_count
    """)

    print("Connection distribution:\n")
    print(f"{'Connections':<15s} {'Memories':>9s} {'Percentage':>11s}")
    print("-" * 70)

    total = 0
    rows = cursor.fetchall()

    for row in rows:
        total += row['memory_count']

    for row in rows:
        conn_count = row['connection_count']
        mem_count = row['memory_count']
        pct = 100 * mem_count / total
        bar = "#" * int(pct)

        label = f"{conn_count} links" if conn_count else "Isolated"
        print(f"{label:<15s} {mem_count:9d} {pct:10.1f}% {bar}")

    print()

    # Graph health score
    cursor.execute("""
    SELECT AVG(
        CASE
            WHEN related_memories = '[]' OR related_memories IS NULL THEN 0
            ELSE json_array_length(related_memories)
        END
    ) as avg_connections
    FROM memories
    """)

    avg_conn = cursor.fetchone()['avg_connections']
    print(f"Average connections per memory: {avg_conn:.2f}")
    print(f"Graph health score: {min(100, avg_conn * 20):.0f}/100")
    print()

    db.close()

def main():
    """Run all visualizations"""
    print()
    print("=" * 70)
    print("           MEMORY INSIGHTS VISUALIZATION")
    print("=" * 70)
    print()

    try:
        visualize_importance_vs_access()
        visualize_access_distribution()
        visualize_category_health()
        visualize_temporal_bursts()
        visualize_tag_insights()
        visualize_graph_structure()

        print("=" * 70)
        print("VISUALIZATION COMPLETE")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
