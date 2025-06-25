import importlib
from pathlib import Path

import pytest


class _DummyRow(dict):
    """row['uprnValue'] / row['contentUrl'] lookup just like ResultRow"""

    def __getitem__(self, key):
        return super().get(key)


class _DummyEndpoint:
    """Replaces SPARQLStore instance inside each script."""

    def __init__(self, rows):
        self._rows = rows

    def query(self, *_):
        return self._rows


def _fake_response():
    class _R:
        status_code = 200
        headers = {"Content-Disposition": 'attachment; filename="file.bin"'}
        content = b"DUMMY"

        def raise_for_status(self):  # noqa: D401
            pass

    return _R()


@pytest.mark.parametrize(
    "mod_name, expects_uprn_subfolder",
    [
        ("examples.get_all_assets_for_a_list_of_uprns", True),
        ("examples.get_all_assets_for_a_uprn", False),
        ("examples.get_all_assets_for_a_uprn_made_by_a_sensor", True),
        ("examples.get_all_assets_of_type_for_list_of_uprns", True),
    ],
)
def test_script_downloads(tmp_path, monkeypatch, mod_name, expects_uprn_subfolder):
    """Import the script as a module, monkey-patch, run main(), check the file."""

    # --------------------------------------
    # import AFTER patching httpx so that the module picks up our stub
    import httpx  # noqa: WPS433 (std-lib)

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _fake_response())

    # make sure env-var exists
    monkeypatch.setenv("API_KEY", "UNIT-TEST-KEY")

    # import module under test
    mod = importlib.import_module(mod_name)

    # patch DOWNLOAD_DIR to the temp folder
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", str(tmp_path))

    # patch ResultRow base-class to something simple & fabricate rows
    monkeypatch.setattr(mod, "ResultRow", _DummyRow)

    dummy_rows = [_DummyRow({"uprnValue": "999", "contentUrl": "https://x/y.bin"})]
    # get the endpoint object name (either global `endpoint` or attr inside module)
    if hasattr(mod, "endpoint"):
        monkeypatch.setattr(mod, "endpoint", _DummyEndpoint(dummy_rows))
    else:
        # safety-net for any differently-named variable
        monkeypatch.setattr(mod, "endpoint", _DummyEndpoint(dummy_rows))

    # --------------------------------------
    # run the script's main()
    mod.main()

    # --------------------------------------
    # assert the file has landed where we expect
    if expects_uprn_subfolder:
        expected = Path(tmp_path) / "999" / "file.bin"
    else:
        expected = Path(tmp_path) / "file.bin"

    assert expected.is_file(), f"{mod_name}: expected {expected} to exist"


# ---------------------------------------------------------------------------
# sanity-check the constant-driven SPARQL strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mod_name, substrings",
    [
        ("examples.get_all_assets_for_a_list_of_uprns", ["200003455212", "5045394"]),
        ("examples.get_all_assets_for_a_uprn", ["5045394"]),  # constant UPRN
        (
            "examples.get_all_assets_for_a_uprn_made_by_a_sensor",
            ["5045394", "bess:OusterLidarSensor"],
        ),
        (
            "examples.get_all_assets_of_type_for_list_of_uprns",
            ["did:rgb-image", "lidar-pointcloud-merged"],
        ),
    ],
)
def test_query_contains_expected_literals(mod_name, substrings):
    """Make sure the hard-coded constants really appear in the QUERY string."""
    mod = importlib.import_module(mod_name)
    q = mod.QUERY
    for s in substrings:
        assert s in q, f"{mod_name}: missing {s} in QUERY"
