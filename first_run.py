#!/usr/bin/env python3
"""
First Run Setup for BuildAutomata Memory MCP
Pre-downloads encoder model and validates dependencies before Claude Desktop integration

Run this BEFORE configuring Claude Desktop to avoid timeout issues.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_status(emoji, message):
    """Print status message with emoji"""
    print(f"{emoji} {message}")

def check_python_version():
    """Verify Python version is 3.8+"""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status("❌", f"Python {version_str} detected - NEED Python 3.8+")
        print("\n   Download from: https://www.python.org/downloads/")
        return False

    print_status("✓", f"Python {version_str} detected")
    return True

def check_pip():
    """Verify pip is available"""
    print_header("Checking pip")

    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status("✓", "pip is available")
            return True
    except Exception as e:
        pass

    print_status("❌", "pip not found")
    print("\n   Install pip: python -m ensurepip --upgrade")
    return False

def check_requirements():
    """Check if requirements are installed"""
    print_header("Checking Required Packages")

    requirements = {
        "mcp": "Model Context Protocol SDK",
        "qdrant_client": "Qdrant vector database client (optional)",
        "sentence_transformers": "Sentence embeddings (optional)"
    }

    missing = []
    optional_missing = []

    for package, description in requirements.items():
        try:
            __import__(package)
            print_status("✓", f"{package}: {description}")
        except ImportError:
            if "optional" in description.lower():
                optional_missing.append(package)
                print_status("⚠", f"{package}: {description} - NOT INSTALLED (optional)")
            else:
                missing.append(package)
                print_status("❌", f"{package}: {description} - NOT INSTALLED")

    if missing:
        print("\n   Install required packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    if optional_missing:
        print("\n   Optional packages not installed (system will use SQLite FTS5 fallback):")
        print(f"   pip install {' '.join(optional_missing)}")

    return True

def check_visual_cpp():
    """Check for Visual C++ redistributables (Windows only)"""
    if platform.system() != "Windows":
        return True

    print_header("Checking Visual C++ Redistributables (Windows)")

    # Check common VC++ DLL locations
    system32 = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32"
    vcruntime_dlls = [
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll"
    ]

    found = []
    missing = []

    for dll in vcruntime_dlls:
        if (system32 / dll).exists():
            found.append(dll)
        else:
            missing.append(dll)

    if found:
        print_status("✓", f"Found VC++ DLLs: {', '.join(found)}")

    if missing:
        print_status("⚠", f"Missing VC++ DLLs: {', '.join(missing)}")
        print("\n   These are needed for sentence-transformers (torch dependency)")
        print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   (System will still work with SQLite FTS5 if you skip this)")
        return False

    return True

def download_encoder():
    """Pre-download sentence transformer encoder model"""
    print_header("Downloading Sentence Encoder Model")

    try:
        from sentence_transformers import SentenceTransformer
        print_status("⏳", "Downloading all-MiniLM-L6-v2 encoder (first time only)...")
        print("   This may take 2-5 minutes depending on connection speed...")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        print_status("✓", "Encoder model downloaded successfully")

        # Test encode
        test_embedding = model.encode("test")
        print_status("✓", f"Encoder working (dimension: {len(test_embedding)})")

        return True
    except ImportError:
        print_status("⚠", "sentence-transformers not installed - skipping encoder download")
        print("   System will use SQLite FTS5 for text search (still works fine)")
        return True
    except Exception as e:
        print_status("❌", f"Encoder download failed: {e}")
        print("\n   You can retry later or use SQLite FTS5 fallback (system still works)")
        return False

def check_sqlite():
    """Verify SQLite3 is available"""
    print_header("Checking SQLite3")

    try:
        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        conn.close()

        print_status("✓", f"SQLite version {version} available")
        return True
    except Exception as e:
        print_status("❌", f"SQLite error: {e}")
        return False

def check_qdrant():
    """Check if Qdrant server is running (optional)"""
    print_header("Checking Qdrant (Optional)")

    try:
        from qdrant_client import QdrantClient

        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))

        print_status("⏳", f"Checking Qdrant at {host}:{port}...")

        client = QdrantClient(host=host, port=port, timeout=3)
        collections = client.get_collections()

        print_status("✓", f"Qdrant server running (found {len(collections.collections)} collections)")
        return True

    except ImportError:
        print_status("⚠", "qdrant-client not installed (optional - system works without it)")
        return True
    except Exception as e:
        print_status("⚠", f"Qdrant not running: {e}")
        print("\n   This is OPTIONAL - system will use SQLite FTS5 for search")
        print("   To use Qdrant (enhanced semantic search):")
        print("   1. Download from: https://github.com/qdrant/qdrant/releases")
        print("   2. Extract qdrant.exe")
        print("   3. Run: qdrant.exe")
        return True

def test_mcp_server():
    """Test that MCP server can initialize"""
    print_header("Testing MCP Server Initialization")

    try:
        print_status("⏳", "Importing buildautomata_memory_mcp module...")

        # Add current directory to path if needed
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        # Import should trigger encoder download if needed
        import buildautomata_memory_mcp

        print_status("✓", "MCP server module loaded successfully")
        return True

    except Exception as e:
        print_status("❌", f"MCP server initialization failed: {e}")
        import traceback
        print("\n   Full error:")
        print(traceback.format_exc())
        return False

def print_next_steps():
    """Print instructions for Claude Desktop configuration"""
    print_header("Setup Complete!")

    print("\n✓ All checks passed - ready for Claude Desktop integration")
    print("\n" + "=" * 70)
    print("  NEXT STEPS - Configure Claude Desktop")
    print("=" * 70)

    print("\n1. Open Claude Desktop")
    print("2. Go to Settings → Developer → Edit Config")
    print("3. Add this to claude_desktop_config.json:\n")

    script_path = Path(__file__).parent / "buildautomata_memory_mcp.py"

    # Windows path formatting
    if platform.system() == "Windows":
        path_str = str(script_path).replace("\\", "/")
        print('   {')
        print('     "mcpServers": {')
        print('       "buildautomata-memory": {')
        print('         "command": "python",')
        print(f'         "args": ["{path_str}"]')
        print('       }')
        print('     }')
        print('   }')
    else:
        print('   {')
        print('     "mcpServers": {')
        print('       "buildautomata-memory": {')
        print('         "command": "python",')
        print(f'         "args": ["{script_path}"]')
        print('       }')
        print('     }')
        print('   }')

    print("\n4. Restart Claude Desktop")
    print("5. Test with: 'Store this memory: Testing memory system'")
    print("\n" + "=" * 70)

def main():
    """Run all checks"""
    print("\n" + "=" * 70)
    print("  BuildAutomata Memory MCP - First Run Setup")
    print("  Copyright 2025 Jurden Bruce")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("pip", check_pip),
        ("Required Packages", check_requirements),
    ]

    # Platform-specific checks
    if platform.system() == "Windows":
        checks.append(("Visual C++ Redistributables", check_visual_cpp))

    checks.extend([
        ("SQLite3", check_sqlite),
        ("Qdrant Server", check_qdrant),
        ("Encoder Model", download_encoder),
        ("MCP Server", test_mcp_server),
    ])

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_status("❌", f"Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    required_checks = ["Python Version", "pip", "Required Packages", "SQLite3", "MCP Server"]
    optional_checks = ["Visual C++ Redistributables", "Qdrant Server", "Encoder Model"]

    required_passed = all(results.get(check, False) for check in required_checks if check in results)
    optional_passed = sum(results.get(check, False) for check in optional_checks if check in results)

    if required_passed:
        print_status("✓", f"REQUIRED: All critical checks passed")
    else:
        print_status("❌", f"REQUIRED: Some critical checks failed - fix these first")

    print_status("ℹ", f"OPTIONAL: {optional_passed}/{len([c for c in optional_checks if c in results])} optional features available")

    if required_passed:
        print_next_steps()
        return 0
    else:
        print("\n⚠  Fix the failed checks above, then run this script again")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
