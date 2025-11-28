# BuildAutomata Memory System - Test Suite

Comprehensive test suite for the BuildAutomata Memory System. Tests all MCP tools end-to-end to ensure system functionality.

## Quick Start

From the project root directory:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose
```

Or directly from the tests directory:

```bash
# Run all tests
python tests/test_memory_system.py

# Run with verbose output
python tests/test_memory_system.py --verbose
```

## What's Tested

### Memory Storage Operations
- ✓ `store_memory` - Basic memory creation
- ✓ `get_memory` - Retrieval by ID
- ✓ `search_memories` - Semantic + FTS search
- ✓ `update_memory` - Memory updates and versioning

### Statistics & Analytics
- ✓ `get_statistics` - System-wide statistics
- ✓ `list_categories` - Category enumeration
- ✓ `list_tags` - Tag enumeration
- ✓ `get_most_accessed_memories` - Access pattern analysis
- ✓ `get_least_accessed_memories` - Dead weight detection

### Session Management
- ✓ `get_session_memories` - Session-based retrieval
- ✓ `get_memory_timeline` - Timeline with diffs and patterns

### Intentions (Proactive Agency)
- ✓ `store_intention` - Intention creation
- ✓ `get_active_intentions` - Active intention retrieval
- ✓ `update_intention_status` - Status updates

### Graph Operations
- ✓ `traverse_memory_graph` - Graph traversal
- ✓ `find_memory_clusters` - Cluster detection
- ✓ `get_graph_stats` - Graph statistics

### System Operations
- ✓ `get_command_history` - Command audit trail
- ✓ `run_maintenance` - Database maintenance
- ✓ `initialize_agent` - Agent initialization

## Test Architecture

### Isolation
- Each test run creates a temporary database
- Tests are completely isolated from production data
- Automatic cleanup after test completion

### Dependencies
- Tests run without Qdrant (SQLite-only mode)
- No external service dependencies
- Self-contained and portable

### Test Data
- Uses unique session ID per test run
- Creates minimal test memories
- Cleans up automatically

## When to Run Tests

Run tests when:
- ✓ Making changes to core functionality
- ✓ Refactoring code
- ✓ Before committing changes
- ✓ After pulling updates
- ✓ When debugging issues
- ✓ Before releases

## Expected Output

### Success
```
======================================================================
BuildAutomata Memory System - Comprehensive Test Suite
======================================================================

=== SETUP ===
Test database: /tmp/memory_test_abc123
Setup complete

[... test execution ...]

=== TEARDOWN ===
Removed test database: /tmp/memory_test_abc123
Teardown complete

======================================================================
RESULTS
======================================================================
Passed: 23
Failed: 0
Total:  23

✓ ALL TESTS PASSED
```

### Failure
```
======================================================================
RESULTS
======================================================================
Passed: 20
Failed: 3
Total:  23

✗ 3 TEST(S) FAILED
```

When tests fail, check the output for specific failure messages indicating which tests failed and why.

## Verbose Mode

Use `--verbose` flag for detailed test output:

```bash
python run_tests.py --verbose
```

This shows:
- Each individual test assertion
- Test execution flow
- Detailed setup/teardown information

## Adding New Tests

To add new tests:

1. Add a new test method to `TestMemorySystem` class in `test_memory_system.py`
2. Follow naming convention: `test_<feature_name>`
3. Use `self.assert_true()` for assertions
4. Add to `run_all_tests()` method in appropriate order
5. Run tests to verify

Example:

```python
def test_new_feature(self):
    """Test description"""
    self.log("\n--- Test: new_feature ---")

    result = self.memory_store.new_feature()

    self.assert_true(
        "expected_key" in result,
        "new_feature returns expected_key"
    )
```


## Continuous Integration

For CI/CD pipelines:

```bash
# Exit code 0 = all tests passed
# Exit code 1 = tests failed
python run_tests.py
```

## Troubleshooting

### ImportError
If you get import errors:
- Ensure you're running from project root
- Check Python path includes parent directory

### Database Errors
If you get database errors:
- Check disk space for temporary database
- Verify write permissions in temp directory
- Check SQLite is available

### Permission Errors
If you get permission errors on Windows:
- Run as administrator if needed
- Check antivirus isn't blocking temp directory access

## License

Copyright 2025 Jurden Bruce

Part of the BuildAutomata Memory System.
