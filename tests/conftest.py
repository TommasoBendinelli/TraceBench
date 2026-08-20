import pytest


def _matlab_engine_available() -> bool:
    try:
        import matlab.engine  # type: ignore

        start_matlab = getattr(matlab.engine, "start_matlab", None)
        return callable(start_matlab)
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running tests")
    config.addinivalue_line(
        "markers",
        "requires_matlab: needs a working MATLAB engine installation",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    matlab_available = _matlab_engine_available()
    timeout = config.getoption("timeout", default=None)
    if timeout is None:
        timeout_value = None
    else:
        timeout_text = str(timeout).strip()
        if not timeout_text:
            timeout_value = None
        else:
            normalized = timeout_text.replace(".", "", 1)
            timeout_value = float(timeout_text) if normalized.isdigit() else None

    skip_matlab = pytest.mark.skip(reason="Skipped MATLAB-dependent tests: MATLAB engine not available.")
    skip_slow = pytest.mark.skip(reason="Skipped slow tests with --timeout=1.")

    for item in items:
        if not matlab_available and "requires_matlab" in item.keywords:
            item.add_marker(skip_matlab)
        if timeout_value is not None and timeout_value <= 1 and "slow" in item.keywords:
            item.add_marker(skip_slow)
