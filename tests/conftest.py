def pytest_configure(config):
    config.addinivalue_line("markers", "slow: torch.compile and other long-running checks")
