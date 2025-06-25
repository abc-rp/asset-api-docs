"""
Unit-tests for query_assist.py
Run with:   pytest -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import query_assist as qa

# ---------- pure helpers -------------------------------------------------- #


@pytest.mark.parametrize(
    "iri, expected",
    [
        ("did:lidar-pointcloud-merged", "lidar-pointcloud-merged"),
        ("https://w3id.org/dob/voc/did:rgb-image", "rgb-image"),
        ("did:ir-false-color-image", "ir-false-color-image"),
    ],
)
def test_asset_subdir(iri, expected):
    """`asset_subdir` should strip `did:` and leave only safe chars."""
    assert qa.asset_subdir(iri) == expected


def test_build_asset_query_filters_are_injected():
    """
    The generated SPARQL must contain the sensor/type filters that
    the CLI flags would inject.
    """
    args = SimpleNamespace(sensor="bess:OusterLidarSensor", types="did:rgb-image")
    query = qa.build_asset_query(["123"], args)
    assert "bess:OusterLidarSensor" in query  # sensor filter
    assert "did:rgb-image" in query  # type filter
    assert '"123"' in query  # UPRN filter


# ---------- download path logic ------------------------------------------- #


def _dummy_response(tmp_path):
    class _Resp:
        status_code = 200
        headers = {"Content-Disposition": 'attachment; filename="file.bin"'}
        content = b"UNIT-TEST"

        def raise_for_status(self):  # noqa: D401
            pass

    return _Resp()


def test_download_flow_creates_nested_dirs(tmp_path, monkeypatch):
    """
    Integration-style test: simulate `--uprn 42` download with one RGB asset.
    Ensures that the target folder becomes `<tmp>/downloads/42/rgb-image/file.bin`.
    """

    # ---- stub external dependencies --------------------------------------
    class DummyStore:
        def __init__(self, *a, **k):
            pass

        def query(self, *_):
            return [
                {
                    "uprnValue": "42",
                    "contentUrl": "https://example.com/file.bin",
                    "enum": "did:rgb-image",
                }
            ]

    monkeypatch.setattr(qa, "SPARQLStore", DummyStore)
    monkeypatch.setattr(qa.httpx, "get", lambda *a, **k: _dummy_response(tmp_path))
    monkeypatch.setenv("API_KEY", "DUMMY")

    # ---- invoke CLI entry-point ------------------------------------------
    argv_backup, sys.argv = sys.argv, [
        "query_assist",
        "--uprn",
        "42",
        "--download-dir",
        str(tmp_path),
    ]
    try:
        qa.main()
    finally:
        sys.argv = argv_backup

    # ---- assert folder structure -----------------------------------------
    target = Path(tmp_path) / "42" / "rgb-image" / "file.bin"
    assert target.is_file(), f"expected {target} to be written"


# ---------- CSV helper ---------------------------------------------------- #


def test_load_column_from_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("uprn\n1\n2\n\n3\n")
    assert qa.load_column_from_csv(csv_path, "uprn") == ["1", "2", "3"]
