"""Unit tests for file utilities."""

from dctools.utilities.file_utils import (
    FileCacheManager,
    check_valid_files,
    empty_folder,
    get_list_filter_files,
    list_files_with_extension,
    load_config_file,
    read_file_tolist,
    remove_file,
)


def test_remove_file_happy_path(tmp_path):
    """remove_file should delete an existing file."""
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")

    assert remove_file(str(f)) is True
    assert not f.exists()


def test_empty_folder_filters_by_extension(tmp_path):
    """empty_folder should delete only matching extensions."""
    (tmp_path / "a.nc").write_text("1", encoding="utf-8")
    (tmp_path / "b.nc").write_text("2", encoding="utf-8")
    (tmp_path / "c.txt").write_text("3", encoding="utf-8")

    deleted = empty_folder(str(tmp_path), extension=".nc")

    assert deleted == 2
    assert (tmp_path / "c.txt").exists()


def test_list_and_filter_files(tmp_path):
    """Listing/filtering helpers should return deterministic matching files."""
    (tmp_path / "alpha_1.nc").write_text("1", encoding="utf-8")
    (tmp_path / "alpha_2.nc").write_text("2", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("3", encoding="utf-8")

    files = list_files_with_extension(str(tmp_path), ".nc")
    prefixed = get_list_filter_files(str(tmp_path), ".nc", "alpha", prefix=True)

    assert files == ["alpha_1.nc", "alpha_2.nc"]
    assert prefixed == ["alpha_1.nc", "alpha_2.nc"]


def test_read_file_tolist_and_check_valid_files(tmp_path):
    """Reading and file validation helpers should behave as expected."""
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    lines = read_file_tolist(str(f), max_lines=2)
    valid = check_valid_files([str(f), str(tmp_path / "missing.txt")])

    assert lines == ["a", "b"]
    assert valid == [str(f)]


def test_load_config_file_and_cache_manager(tmp_path):
    """YAML loading and cache manager eviction should both work."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1\n", encoding="utf-8")
    loaded = load_config_file(str(cfg))
    assert loaded["a"] == 1

    f1 = tmp_path / "f1.tmp"
    f2 = tmp_path / "f2.tmp"
    f1.write_text("1", encoding="utf-8")
    f2.write_text("2", encoding="utf-8")

    cache = FileCacheManager(max_files=1)
    cache.add(str(f1))
    cache.add(str(f2))

    assert not f1.exists()
    assert f2.exists()
    cache.clear()
    assert not f2.exists()
