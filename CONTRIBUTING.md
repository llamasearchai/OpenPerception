# Contributing to OpenPerception

First off, thank you for considering contributing to OpenPerception! It's people like you that make OpenPerception such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please report unacceptable behavior to nikjois@llamasearch.ai.

## How Can I Contribute?

### Reporting Bugs

* **Check if the bug has already been reported** by searching on GitHub under [Issues](https://github.com/llamasearchai/OpenPerception/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/llamasearchai/OpenPerception/issues/new). Be sure to include:
  * A clear title and description
  * As much relevant information as possible
  * A code sample or test case demonstrating the unexpected behavior

### Suggesting Enhancements

* Open a new issue with a clear title and detailed description.
* Provide specific examples and steps for implementing the enhancement.
* Explain why this enhancement would be useful to most OpenPerception users.

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the existing style guidelines.
6. Issue your pull request!

## Development Workflow

1. Clone your fork of the repository.
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Make your changes.
4. Run tests:
   ```bash
   pytest tests/
   ```
5. Submit your pull request.

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [PEP 257](https://www.python.org/dev/peps/pep-0257/)
* Use 4 spaces for indentation
* Use docstrings for all classes and methods
* Use type hints (PEP 484) for function arguments and return values

### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown) for documentation.
* Reference code using backticks.
* Provide examples when appropriate.

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Indicates an unexpected problem or unintended behavior
* `documentation`: Improvements or additions to documentation
* `enhancement`: New feature or request
* `help-wanted`: Extra attention is needed
* `good-first-issue`: Good for newcomers

## Thank You!

Your contributions to open source, large or small, make projects like this possible. Thank you for taking the time to contribute. 