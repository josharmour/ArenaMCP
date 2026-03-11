import json

from arenamcp.scryfall import ScryfallCache


def _new_cache(tmp_path):
    cache = ScryfallCache.__new__(ScryfallCache)
    cache._cache_dir = tmp_path
    cache._arena_index = {}
    cache._last_api_call = 0.0
    cache._not_found_cache = set()
    cache._name_cache = {}
    return cache


def test_load_or_download_recovers_from_corrupt_cache(monkeypatch, tmp_path):
    cache = _new_cache(tmp_path)
    bulk_path = tmp_path / "default_cards.json"
    bulk_path.write_text("{broken", encoding="utf-8")

    calls = {"load": 0, "download": 0}

    monkeypatch.setattr(cache, "_is_cache_stale", lambda: False)

    def _fake_load():
        calls["load"] += 1
        if calls["load"] == 1:
            raise json.JSONDecodeError("bad json", "{broken", 1)
        cache._arena_index = {123: {"arena_id": 123, "name": "Recovered"}}

    def _fake_download():
        calls["download"] += 1
        bulk_path.write_text('[{"arena_id": 123, "name": "Recovered"}]', encoding="utf-8")

    monkeypatch.setattr(cache, "_load_bulk_data", _fake_load)
    monkeypatch.setattr(cache, "_download_bulk_data", _fake_download)

    cache._load_or_download_bulk_data()

    assert calls["load"] == 2
    assert calls["download"] == 1
    assert 123 in cache._arena_index


def test_load_or_download_does_not_raise_when_recovery_fails(monkeypatch, tmp_path):
    cache = _new_cache(tmp_path)
    bulk_path = tmp_path / "default_cards.json"
    bulk_path.write_text("{still-broken", encoding="utf-8")

    monkeypatch.setattr(cache, "_is_cache_stale", lambda: False)
    monkeypatch.setattr(
        cache,
        "_load_bulk_data",
        lambda: (_ for _ in ()).throw(json.JSONDecodeError("bad json", "{", 1)),
    )
    monkeypatch.setattr(
        cache,
        "_download_bulk_data",
        lambda: (_ for _ in ()).throw(RuntimeError("download unavailable")),
    )

    # Should degrade gracefully with empty index, not raise.
    cache._load_or_download_bulk_data()
    assert cache._arena_index == {}
