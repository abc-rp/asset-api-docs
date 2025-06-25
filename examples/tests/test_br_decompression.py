import br_decompress as br
import brotli
import query_assist as qa


def _make_compressed_pair():
    """Return (raw_bytes, brotli_compressed_bytes)."""
    raw = b"FOR UNIT TEST ONLY - pretend this is a PCD header\n"
    return raw, brotli.compress(raw)


class _DummyResponse:
    def __init__(self, data):
        self.status_code = 200
        self.headers = {"Content-Disposition": 'attachment; filename="cloud.pcd.br"'}
        self.content = data

    def raise_for_status(self):
        pass


class _DummyStore:
    """Fake rdflib SPARQLStore that yields exactly one result row."""

    def __init__(self, *_, **__):
        pass

    def query(self, *_):
        return [
            {
                "uprnValue": "999",
                "contentUrl": "https://example.com/cloud.pcd.br",
                "enum": "did:lidar-pointcloud-merged",
            }
        ]


def test_download_and_decompress_brotli(tmp_path, monkeypatch):
    raw, compressed = _make_compressed_pair()

    monkeypatch.setattr(qa, "SPARQLStore", _DummyStore)

    monkeypatch.setattr(qa.httpx, "get", lambda *a, **k: _DummyResponse(compressed))

    monkeypatch.setenv("API_KEY", "DUMMY")

    monkeypatch.setattr(
        qa,
        "parse_args",
        lambda: qa.argparse.Namespace(
            uprn=["999"],
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

    p_br = tmp_path / "999" / "lidar-pointcloud-merged" / "cloud.pcd.br"
    assert p_br.is_file(), "compressed asset should have been saved by query_assist"

    br.find_and_replace_pcd_br(str(tmp_path))

    p_raw = p_br.with_suffix("")
    assert p_raw.is_file(), "decompressed .pcd should exist"
    assert not p_br.exists(), ".pcd.br should have been removed"

    assert p_raw.read_bytes() == raw, "decompressed bytes should match original"
