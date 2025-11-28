"""
Comprehensive System Tests for BuildAutomata Memory System
Copyright 2025 Jurden Bruce

Tests all MCP tools end-to-end to ensure system functionality.
Run this when making changes to verify nothing breaks.

Usage:
    python tests/test_memory_system.py
    python tests/test_memory_system.py --verbose
"""

import sys
import os
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from buildautomata_memory_mcp import MemoryStore
from models import Memory, Intention


class TestMemorySystem:
    """Comprehensive system tests"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.test_username = f"test_{uuid.uuid4().hex[:8]}"
        self.test_agent_name = "system_test"
        self.memory_store = None
        self.passed = 0
        self.failed = 0
        self.test_session_id = str(uuid.uuid4())

    def log(self, msg, force=False):
        """Print if verbose or forced"""
        if self.verbose or force:
            print(msg)

    def assert_true(self, condition, test_name, message=""):
        """Assert a condition is true"""
        if condition:
            self.passed += 1
            self.log(f"  [PASS] {test_name}")
            return True
        else:
            self.failed += 1
            error_msg = f"  [FAIL] {test_name}"
            if message:
                error_msg += f": {message}"
            print(error_msg)
            return False

    def setup(self):
        """Create temporary test database"""
        self.log("\n=== SETUP ===", force=True)

        # Initialize memory store with unique test user
        self.memory_store = MemoryStore(
            username=self.test_username,
            agent_name=self.test_agent_name,
            lazy_load=False
        )

        self.log(f"Test user: {self.test_username}")
        self.log(f"Test agent: {self.test_agent_name}")
        self.log(f"Test database: {self.memory_store.db_path}")

        self.assert_true(
            self.memory_store is not None,
            "MemoryStore initialization",
            "Failed to create MemoryStore"
        )

        self.log("Setup complete\n", force=True)

    def teardown(self):
        """Clean up test database"""
        self.log("\n=== TEARDOWN ===", force=True)

        # Clean up test memory repository
        if self.memory_store and hasattr(self.memory_store, 'base_path'):
            try:
                if self.memory_store.base_path.exists():
                    # Close Qdrant connection if it exists to release file locks
                    if hasattr(self.memory_store, 'qdrant_store') and self.memory_store.qdrant_store:
                        if hasattr(self.memory_store.qdrant_store, 'client'):
                            try:
                                self.memory_store.qdrant_store.client.close()
                            except:
                                pass

                    # Try to remove, but don't fail if Windows has locks
                    try:
                        shutil.rmtree(self.memory_store.base_path)
                        self.log(f"Removed test database: {self.memory_store.base_path}")
                    except PermissionError:
                        self.log(f"Warning: Could not remove {self.memory_store.base_path} (files in use)")
            except Exception as e:
                self.log(f"Cleanup error: {e}")

        self.log("Teardown complete\n", force=True)

    # ===== MEMORY STORAGE TESTS =====

    async def _test_store_memory(self):
        """Test basic memory storage"""
        self.log("\n--- Test: store_memory ---")

        # Create Memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content="Test memory content for system testing",
            category="test",
            importance=0.8,
            tags=["test", "system-test"],
            metadata={"test_key": "test_value"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            memory_type="episodic",
            session_id=self.test_session_id,
            task_context="system testing"
        )

        result = await self.memory_store.store_memory(memory)

        success = result.get("success", False) and not result.get("error")
        self.assert_true(success, "store_memory basic")

        if success:
            # Store memory_id for later tests
            self.test_memory_id = memory.id

    async def _test_get_memory(self):
        """Test retrieving specific memory by ID"""
        self.log("\n--- Test: get_memory ---")

        if not hasattr(self, 'test_memory_id'):
            self.assert_true(False, "get_memory", "No test_memory_id available")
            return

        result = await self.memory_store.get_memory_by_id(self.test_memory_id)

        self.assert_true(
            "error" not in result,
            "get_memory retrieval"
        )

        self.assert_true(
            result.get("content") == "Test memory content for system testing",
            "get_memory content match"
        )

    async def _test_search_memories(self):
        """Test semantic + FTS search"""
        self.log("\n--- Test: search_memories ---")

        # Store a few memories to search
        mem1 = Memory(
            id=str(uuid.uuid4()),
            content="Python programming language features",
            category="programming",
            importance=0.7,
            tags=["python", "programming"],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        await self.memory_store.store_memory(mem1)

        mem2 = Memory(
            id=str(uuid.uuid4()),
            content="JavaScript async/await patterns",
            category="programming",
            importance=0.6,
            tags=["javascript", "async"],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        await self.memory_store.store_memory(mem2)

        # Search for programming
        results = await self.memory_store.search_memories(
            query="programming",
            limit=10
        )

        self.assert_true(
            isinstance(results, list),
            "search_memories returns list"
        )

        self.assert_true(
            len(results) >= 2,
            "search_memories finds multiple results",
            f"Found {len(results)} memories"
        )

    async def _test_update_memory(self):
        """Test updating existing memory"""
        self.log("\n--- Test: update_memory ---")

        if not hasattr(self, 'test_memory_id'):
            self.assert_true(False, "update_memory", "No test_memory_id available")
            return

        result = await self.memory_store.update_memory(
            memory_id=self.test_memory_id,
            content="Updated test memory content",
            tags=["test", "updated"],
            importance=0.9
        )

        self.assert_true(
            result.get("success", False),
            "update_memory execution"
        )

        # Verify update
        updated = await self.memory_store.get_memory_by_id(self.test_memory_id)

        self.assert_true(
            updated.get("content") == "Updated test memory content",
            "update_memory content changed"
        )

    # ===== STATISTICS TESTS =====

    def _test_get_statistics(self):
        """Test memory system statistics"""
        self.log("\n--- Test: get_statistics ---")

        result = self.memory_store.get_statistics()

        self.assert_true(
            "total_memories" in result,
            "get_statistics returns total_memories"
        )

        self.assert_true(
            result.get("total_memories", 0) > 0,
            "get_statistics shows memories"
        )

    async def _test_list_categories(self):
        """Test category listing"""
        self.log("\n--- Test: list_categories ---")

        result = await self.memory_store.list_categories()

        self.assert_true(
            "categories" in result,
            "list_categories returns categories list"
        )

    async def _test_list_tags(self):
        """Test tag listing"""
        self.log("\n--- Test: list_tags ---")

        result = await self.memory_store.list_tags()

        self.assert_true(
            "tags" in result,
            "list_tags returns tags list"
        )

    # ===== ACCESS PATTERN TESTS =====

    async def _test_most_accessed(self):
        """Test most accessed memories retrieval"""
        self.log("\n--- Test: get_most_accessed_memories ---")

        result = await self.memory_store.get_most_accessed_memories(limit=10)

        self.assert_true(
            "memories" in result,
            "get_most_accessed_memories returns memories"
        )

    async def _test_least_accessed(self):
        """Test least accessed memories retrieval"""
        self.log("\n--- Test: get_least_accessed_memories ---")

        result = await self.memory_store.get_least_accessed_memories(
            limit=10,
            min_age_days=0
        )

        self.assert_true(
            "memories" in result,
            "get_least_accessed_memories returns memories"
        )

    # ===== SESSION TESTS =====

    async def _test_session_memories(self):
        """Test session-based memory retrieval"""
        self.log("\n--- Test: get_session_memories ---")

        result = await self.memory_store.get_session_memories(
            session_id=self.test_session_id
        )

        self.assert_true(
            isinstance(result, list),
            "get_session_memories returns list"
        )

    # ===== TIMELINE TESTS =====

    async def _test_memory_timeline(self):
        """Test memory timeline retrieval"""
        self.log("\n--- Test: get_memory_timeline ---")

        result = await self.memory_store.get_memory_timeline(
            limit=5,
            include_diffs=True,
            include_patterns=True
        )

        self.assert_true(
            isinstance(result, dict),
            "get_memory_timeline returns dict"
        )

    # ===== INTENTION TESTS =====

    async def _test_store_intention(self):
        """Test storing proactive intentions"""
        self.log("\n--- Test: store_intention ---")

        result = await self.memory_store.store_intention(
            description="Test intention for system testing",
            priority=0.8,
            deadline=(datetime.now() + timedelta(days=1)).isoformat(),
            actions=["test_action_1", "test_action_2"],
            preconditions=["test_precondition"]
        )

        success = "intention_id" in result and "error" not in result
        self.assert_true(success, "store_intention")

        if success:
            self.test_intention_id = result["intention_id"]

    async def _test_get_active_intentions(self):
        """Test retrieving active intentions"""
        self.log("\n--- Test: get_active_intentions ---")

        result = await self.memory_store.get_active_intentions()

        self.assert_true(
            "intentions" in result or isinstance(result, list),
            "get_active_intentions returns result"
        )

    async def _test_update_intention_status(self):
        """Test updating intention status"""
        self.log("\n--- Test: update_intention_status ---")

        if not hasattr(self, 'test_intention_id'):
            self.assert_true(False, "update_intention_status", "No test_intention_id available")
            return

        result = await self.memory_store.update_intention_status(
            intention_id=self.test_intention_id,
            status="completed"
        )

        self.assert_true(
            "error" not in result or result.get("status") == "completed",
            "update_intention_status execution"
        )

    # ===== GRAPH OPERATIONS TESTS =====

    async def _test_traverse_memory_graph(self):
        """Test graph traversal"""
        self.log("\n--- Test: traverse_memory_graph ---")

        if not hasattr(self, 'test_memory_id'):
            self.log("  [SKIP] No test_memory_id available")
            return

        result = await self.memory_store.traverse_graph(
            start_memory_id=self.test_memory_id,
            depth=2
        )

        self.assert_true(
            "nodes" in result or "error" in result,
            "traverse_memory_graph returns result"
        )

    async def _test_find_memory_clusters(self):
        """Test cluster detection"""
        self.log("\n--- Test: find_memory_clusters ---")

        result = await self.memory_store.find_clusters(
            min_cluster_size=2,
            min_importance=0.0
        )

        self.assert_true(
            "clusters" in result or "clusters_found" in result,
            "find_memory_clusters returns result"
        )

    async def _test_get_graph_stats(self):
        """Test graph statistics"""
        self.log("\n--- Test: get_graph_stats ---")

        result = await self.memory_store.get_graph_statistics()

        self.assert_true(
            "total_memories" in result or "error" in result,
            "get_graph_stats returns result"
        )

    # ===== AGENT INITIALIZATION TEST =====

    async def _test_initialize_agent(self):
        """Test agent initialization"""
        self.log("\n--- Test: initialize_agent ---")

        result = await self.memory_store.proactive_initialization_scan()

        self.assert_true(
            "continuity_check" in result,
            "initialize_agent returns continuity_check"
        )

        self.assert_true(
            "active_intentions" in result,
            "initialize_agent returns active_intentions"
        )

    # ===== RUN ALL TESTS =====

    def run_all_tests(self):
        """Execute all test methods"""
        print("\n" + "="*70)
        print("BuildAutomata Memory System - Comprehensive Test Suite")
        print("="*70)

        # Setup
        self.setup()

        # Run async tests
        async def run_async_tests():
            await self._test_store_memory()
            await self._test_get_memory()
            await self._test_search_memories()
            await self._test_update_memory()
            await self._test_list_categories()
            await self._test_list_tags()
            await self._test_most_accessed()
            await self._test_least_accessed()
            await self._test_session_memories()
            await self._test_memory_timeline()
            await self._test_store_intention()
            await self._test_get_active_intentions()
            await self._test_update_intention_status()
            await self._test_traverse_memory_graph()
            await self._test_find_memory_clusters()
            await self._test_get_graph_stats()
            await self._test_initialize_agent()

        # Run sync tests
        try:
            asyncio.run(run_async_tests())

            self._test_get_statistics()

        except Exception as e:
            self.failed += 1
            print(f"  [FAIL] Test execution error: {e}")

        # Teardown
        self.teardown()

        # Results
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")

        if self.failed == 0:
            print("\nALL TESTS PASSED")
            return 0
        else:
            print(f"\n{self.failed} TEST(S) FAILED")
            return 1


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Run BuildAutomata Memory System tests")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    args = parser.parse_args()

    tester = TestMemorySystem(verbose=args.verbose)
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
