def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "codeblocks: mark test to be collected from code blocks",
    ) 