# OpenPerception Documentation

This directory contains the source files for the OpenPerception documentation.

## Building the Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org) to generate HTML, PDF, and other formats.

### Prerequisites

Install the required packages:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
```

### Building HTML Documentation

```bash
# Navigate to the docs directory
cd docs

# Build the HTML documentation
make html

# The output will be in _build/html/
```

### Building PDF Documentation

```bash
# Make sure you have LaTeX installed
# On Ubuntu/Debian:
# sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra latexmk

# On macOS (with Homebrew):
# brew install --cask mactex

# On Windows:
# Install MiKTeX from https://miktex.org/download

# Navigate to the docs directory
cd docs

# Build the PDF documentation
make latexpdf

# The output will be in _build/latex/
```

## Documentation Structure

- `index.rst` - Main index page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `user_guide/` - User guide for different modules
- `api/` - API reference documentation
- `examples/` - Examples and tutorials
- `development/` - Development guide for contributors

## Contributing to Documentation

We welcome contributions to improve the documentation! When contributing:

1. Write clear and concise text
2. Use proper reStructuredText or Markdown syntax
3. Include examples where appropriate
4. Check for spelling and grammar errors
5. Test that the documentation builds correctly

## Generating API Documentation

API documentation is automatically generated from docstrings in the code. To update the API documentation:

```bash
sphinx-apidoc -f -o api ../src/openperception
```

## License

The documentation is licensed under the MIT License, same as the OpenPerception framework. 