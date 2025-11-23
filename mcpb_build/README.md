# BuildAutomata Memory MCP - Desktop Extension Build

This directory contains everything needed to build the `.mcpb` Desktop Extension package for one-click installation in Claude Desktop.

## What is a Desktop Extension (.mcpb)?

Desktop Extensions are a packaging format that makes MCP servers installable via single-click in Claude Desktop. No Python installation, no config files, no terminal commands required.

## Current Status

**Waiting for:** Audit and modular refactor (in progress via Claude Code Web)

**Completed Research:**
- ✅ Qdrant embedded mode working - no separate server needed
- ✅ sentence-transformers model: 90MB (download on first run)
- ✅ Performance benchmarked: 3.5ms queries at 1100 memories
- ✅ Zero-config strategy confirmed viable

## Build Strategy

**Primary: Binary Extension**
- Target: Non-technical users
- Package: Standalone executable (PyInstaller/Nuitka)
- User requirements: NONE (includes Python runtime, all dependencies)
- Distribution: Windows .exe, macOS universal binary, Linux AppImage

**Secondary: Python Extension**
- Target: Developers who want to modify code
- Package: Python source + bundled dependencies
- User requirements: Python 3.9+ installed
- Distribution: Source files with manifest.json

## Directory Structure (After Refactor)

```
mcpb_build/
├── README.md                          # This file
├── todo_make_mcpb_extension.md        # Detailed implementation plan
├── manifest.json                      # Extension metadata (TBD)
├── build_scripts/
│   ├── build_binary.sh                # PyInstaller/Nuitka build script
│   ├── build_python.sh                # Python extension packager
│   └── test_extension.sh              # Local testing workflow
├── src/                               # Refactored source (from main repo)
│   ├── models.py
│   ├── storage/
│   ├── cache.py
│   ├── store.py
│   └── mcp_server.py
├── icon.png                           # Extension icon (TBD)
├── screenshots/                       # For directory submission (TBD)
└── dist/                              # Build output (.mcpb files)
    ├── buildautomata_memory.mcpb      # Binary version
    └── buildautomata_memory_dev.mcpb  # Python version
```

## Next Steps

1. **Wait for refactor:** Claude Code Web audit completes, modular code ready
2. **Copy refactored code:** Move modules into `src/`
3. **Create manifest.json:** Define extension metadata, user config, platform support
4. **Build binary:** Compile with PyInstaller/Nuitka
5. **Test on clean VM:** Windows machine with no dev tools
6. **Test on i5 laptop:** Real-world performance validation (32GB DDR4)
7. **Package .mcpb:** Use `mcpb pack` to create final packages
8. **Submit to directory:** (Optional) Submit to Anthropic's extension directory

## Testing Plan

**Phase 1: Development Machine (i9-12900, 64GB DDR5)**
- Binary compilation works
- MCP stdio protocol functional
- Extension loads in Claude Desktop
- All tools accessible

**Phase 2: Clean Windows VM**
- No Python installed
- No dev tools
- Binary runs standalone
- Extension installs one-click

**Phase 3: i5 Laptop (32GB DDR4)**
- Real-world user hardware
- Performance acceptable (<5ms queries)
- First-run setup <5 seconds
- Model download works

**Success Criteria:**
- User drags .mcpb into Claude Desktop
- Clicks "Install"
- Extension works immediately
- No manual configuration required

## Resources

- [MCPB Specification](https://github.com/anthropics/mcpb/blob/main/README.md)
- [Manifest Documentation](https://github.com/anthropics/mcpb/blob/main/MANIFEST.md)
- [Desktop Extensions Announcement](https://www.anthropic.com/engineering/desktop-extensions)
- [PyInstaller Documentation](https://pyinstaller.org/)
- [Nuitka Documentation](https://nuitka.net/)

## License

Copyright (c) 2025 Jurden Bruce

This software uses a custom license allowing:
- **Free use** for personal, educational, and small business (<$100k revenue) purposes
- **Paid licensing** required for companies with $100k+ annual revenue

See [LICENSE](LICENSE) file for full terms.

For commercial licensing: sales@brucepro.net

## Contributing

This is a packaging/distribution directory. Core development happens in the main repo. Once refactor is complete, we'll sync code here for packaging.
