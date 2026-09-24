import pytest


@pytest.fixture(scope="session")
def spark():
    from lakehouse.config import spark_session
    s = spark_session("tests", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture
def lake(tmp_path, monkeypatch):
    """Point every table and the landing zone at a temporary directory."""
    import lakehouse.bronze as bronze
    import lakehouse.config as config
    tables = {k: tmp_path / "lake" / k for k in config.TABLES}
    for mod in (config, bronze):
        monkeypatch.setattr(mod, "LANDING", tmp_path / "landing", raising=False)
    monkeypatch.setattr(config, "TABLES", tables)
    for name in ("bronze", "silver", "quality", "gold", "bench"):
        mod = __import__(f"lakehouse.{name}", fromlist=["TABLES"])
        monkeypatch.setattr(mod, "TABLES", tables, raising=False)
    return tmp_path
