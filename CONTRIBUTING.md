# Contributing to BuildAutomata Memory MCP

Thank you for your interest in contributing! This project welcomes contributions from the community.

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 💬 Help others in discussions

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/buildautomata_memory_mcp_dev.git
   ```
3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Test your changes**
6. **Submit a pull request**

## Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/buildautomata_memory_mcp_dev.git
cd buildautomata_memory_mcp_dev

# Install dependencies
pip install -r requirements.txt

# Optional: Set up Qdrant for testing
docker run -d -p 6333:6333 qdrant/qdrant
```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions and classes
- Comment complex logic
- Keep functions focused and single-purpose

## Testing Your Changes

Before submitting:

```bash
# Test MCP server manually
python buildautomata_memory_mcp.py

# Test CLI
python interactive_memory.py search "test"
python interactive_memory.py store "test memory" --category test
python interactive_memory.py stats

# Verify no errors in logs
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style
- [ ] All tests pass
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Description Should Include

- What changed and why
- Any breaking changes
- Screenshots (if UI-related)
- Testing performed

### Example PR Description

```markdown
## Summary
Added fuzzy search capability to improve query matching

## Changes
- Implemented Levenshtein distance for typo tolerance
- Added --fuzzy flag to search command
- Updated documentation

## Testing
- Tested with various misspellings
- Verified backward compatibility
- No performance impact on exact matches

## Breaking Changes
None
```

## Reporting Bugs

Use GitHub Issues with this template:

**Bug Description:**
Clear description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. See error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: Windows 10 / Mac OS / Linux
- Python version: 3.10.x
- Claude Desktop version: x.x.x
- Qdrant installed: Yes/No

**Logs:**
```
Paste relevant error messages or logs
```

## Suggesting Features

Use GitHub Issues with this template:

**Feature Description:**
Clear description of the feature

**Use Case:**
Why is this useful? What problem does it solve?

**Proposed Solution:**
How might this be implemented?

**Alternatives Considered:**
Other approaches you've thought about

## Areas We Need Help

### High Priority
- [ ] Automated testing suite
- [ ] Memory relationship graphs
- [ ] Web UI for memory management
- [ ] Multi-user support

### Documentation
- [ ] Video tutorials
- [ ] More usage examples
- [ ] API documentation
- [ ] Translation to other languages

### Features
- [ ] Batch import/export
- [ ] Smart auto-tagging
- [ ] Memory consolidation
- [ ] Multi-modal support (images, audio)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Not Acceptable

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

## Questions?

- Check existing [GitHub Issues](https://github.com/yourusername/buildautomata_memory_mcp_dev/issues)
- Start a [Discussion](https://github.com/yourusername/buildautomata_memory_mcp_dev/discussions)
- Review [README.md](README.md) and [SETUP.md](SETUP.md)

## Commercial Use Note

This is an open-source project with a fair-use license. Large companies (>$100k revenue) need a commercial license. See [LICENSE](LICENSE) for details.

If you're contributing on behalf of a large company, please ensure your company has appropriate licensing.

## Recognition

Contributors will be:
- Listed in project credits
- Mentioned in release notes
- Forever appreciated! 🎉

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

Thank you for making BuildAutomata Memory better! 🚀
