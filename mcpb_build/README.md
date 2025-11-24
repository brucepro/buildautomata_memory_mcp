# BuildAutomata Memory MCP - Desktop Extension Build

This directory contains everything needed to build the `.mcpb` Desktop Extension package for one-click installation in Claude Desktop.

## What is a Desktop Extension (.mcpb)?

Desktop Extensions are a packaging format that makes MCP servers installable via single-click in Claude Desktop. No Python installation, no config files, no terminal commands required.

## Instructions
Open Claude Desktop
Drag dist_extension/buildautomata-memory-1.1.0.mcpb to extensions inside of Claude Desktop. Make sure all dependances are installed pip -r requirements.txt


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
