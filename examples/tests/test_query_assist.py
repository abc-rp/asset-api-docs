from types import SimpleNamespace

import pytest
import query_assist as qa


@pytest.mark.parametrize(
    "iri, expected",
    [
        ("did:lidar-pointcloud-merged", "lidar-pointcloud-merged"),
        ("https://w3id.org/foo/did:rgb-image", "rgb-image"),
        ("did:ir-false-color-image", "ir-false-color-image"),
        ("did:weird chars!*£$", "weird_chars____"),  # sanitised
    ],
)
def test_asset_subdir(iri, expected):
    assert qa.asset_subdir(iri) == expected


def test_load_column_from_csv_good(tmp_path):
    csv_file = tmp_path / "uprns.csv"
    csv_file.write_text("uprn\n1\n2\n\n3\n")
    assert qa.load_column_from_csv(csv_file, "uprn") == ["1", "2", "3"]


def test_load_column_from_csv_missing_col(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("not_uprn\n123\n")
    with pytest.raises(RuntimeError, match="missing required 'uprn'"):
        qa.load_column_from_csv(csv_file, "uprn")


def test_build_asset_query_injects_everything():
    args = SimpleNamespace(sensor="bess:OusterLidarSensor", types="did:rgb-image")
    q = qa.build_asset_query(["123"], args)
    assert "bess:OusterLidarSensor" in q
    assert "did:rgb-image" in q
    assert '"123"' in q
    # should ALWAYS bind ?enum
    assert "?enum" in q and "dob:typeQualifier" in q


def test_build_output_area_query_formats_values():
    q = qa.build_output_area_query(["sid:E0001"])
    assert "VALUES ?outputArea { sid:E0001 }" in q


def test_build_ods_to_uprn_query_values_clause():
    q = qa.build_ods_to_uprn_query(["00ABC", "99XYZ"])
    assert 'VALUES ?odsValue { "00ABC" "99XYZ" }' in q


def _dummy_http_response():
    class _R:
        status_code = 200
        headers = {"Content-Disposition": 'attachment; filename="file.bin"'}
        content = b"PSEUDO-BINARY"

        def raise_for_status(self):
            pass

    return _R()


class _DummyStore:
    """Stand-in for rdflib SPARQLStore; returns a synthetic row list."""

    def __init__(self, *a, **k):
        pass

    def query(self, *_):
        class _PhenomenonTime:
            def __init__(self):
                # fixed date for deterministic path
                from datetime import datetime

                self.value = datetime(2024, 1, 2)

        return [
            {
                "uprnValue": "42",
                "contentUrl": "https://example.com/file.bin",
                "enum": "did:rgb-image",
                "phenomenonTime": _PhenomenonTime(),
            }
        ]


def test_cli_download_creates_nested_dir(tmp_path, monkeypatch):
    """Full happy-path run – ensures <download-dir>/<uprn>/<type>/file.bin is created."""
    monkeypatch.setattr(qa, "SPARQLStore", _DummyStore)

    # Mock httpx.Client context manager used in download_asset
    class _DummyClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _dummy_http_response()

    monkeypatch.setattr(qa.httpx, "Client", _DummyClient)
    monkeypatch.setenv("API_KEY", "FAKE-KEY")

    argv = ["query_assist", "--uprn", "42", "--download-dir", str(tmp_path)]
    monkeypatch.setattr(
        qa,
        "parse_args",
        lambda: qa.argparse.Namespace(
            uprn=["42"],
            ods=None,
            sensor=None,
            types=None,
            output_area=None,
            db_url="http://dummy",
            download_dir=str(tmp_path),
            api_key_env="API_KEY",
        ),
    )

    qa.main()

    # Includes date directory now (YYYY-MM-DD)
    expected = tmp_path / "42" / "2024-01-02" / "rgb-image" / "file.bin"
    assert expected.is_file(), f"expected {expected} to exist"


def test_cli_missing_api_key_logs_error(monkeypatch, caplog):
    """Modern behaviour: when API key missing, log error and return without raising."""
    monkeypatch.setattr(qa, "SPARQLStore", _DummyStore)
    monkeypatch.delenv("API_KEY", raising=False)

    monkeypatch.setattr(
        qa,
        "parse_args",
        lambda: qa.argparse.Namespace(
            uprn=["1"],
            ods=None,
            sensor=None,
            types=None,
            output_area=None,
            db_url="http://dummy",
            download_dir=None,
            api_key_env="API_KEY",
        ),
    )
    caplog.clear()
    qa.main()
    assert any(
        "API key environment variable 'API_KEY' is not set." in r.message
        for r in caplog.records
    ), "Expected error log for missing API key"
