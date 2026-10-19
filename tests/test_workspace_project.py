import os
import subprocess

import pytest

from beyondml.workspace.project import PathError, Project, ProjectError


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def test_init_creates_layout(tmp_path):
    p = Project.init(tmp_path)
    for d in ["data/raw", "data/processed", "src", "models", "reports", "agents", ".beyondml/skills"]:
        assert (tmp_path / d).is_dir()
    assert "(not set yet)" in p.read("PROJECT.md")
    assert p.read("MEMORY.md") == "# Memory\n"
    assert (tmp_path / ".beyondml/settings.toml").exists()
    assert len(git(tmp_path, "log", "--oneline").splitlines()) == 1
    assert "data/raw/" in p.read(".gitignore")


def test_init_is_idempotent(tmp_path):
    p = Project.init(tmp_path, goal="g")
    p.write("PROJECT.md", "edited")
    Project.init(tmp_path)
    assert p.read("PROJECT.md") == "edited"
    assert len(git(tmp_path, "log", "--oneline").splitlines()) == 1


def test_open_rejects_plain_folder(tmp_path):
    with pytest.raises(ProjectError):
        Project.open(tmp_path)


def test_path_guard(tmp_path):
    p = Project.init(tmp_path / "proj")
    with pytest.raises(PathError):
        p.path("../x")
    with pytest.raises(PathError):
        p.path(str(tmp_path.anchor) + "Windows")
    with pytest.raises(PathError):
        p.path("data/raw/a.csv", write=True)
    p.path("data/raw/a.csv")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, p.root / "link", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PathError):
        p.path("link/f.txt")


def test_checkpoint_only_on_change(tmp_path):
    p = Project.init(tmp_path)
    p.write("a.txt", "1")
    s1 = p.checkpoint("one")
    assert p.checkpoint("two") == s1


def test_restore_only_given_paths(tmp_path):
    p = Project.init(tmp_path)
    p.write("a.txt", "old")
    sha = p.checkpoint("base")
    p.write("a.txt", "new")
    p.write("b.txt", "b")
    p.write("c.txt", "c")
    p.restore(sha, ["a.txt", "b.txt"])
    assert p.read("a.txt") == "old"
    assert not (tmp_path / "b.txt").exists()
    assert (tmp_path / "c.txt").exists()


def test_memory_capped(tmp_path):
    p = Project.init(tmp_path)
    for i in range(250):
        p.append_memory(str(i))
    lines = p.read("MEMORY.md").splitlines()
    assert lines[0] == "# Memory" and len(lines) == 201
    assert "249" in lines and "0" not in lines


def test_settings_merge(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text('[model]\ndefault = "x"\n')
    monkeypatch.setenv("BEYONDML_HOME", str(home))
    p = Project.init(tmp_path / "proj")
    p.write(".beyondml/settings.toml", '[run]\nmode = "ask"\n')
    s = p.settings()
    assert s["model"]["default"] == "x"
    assert s["model"]["api_base"] == "http://127.0.0.1:11434"
    assert s["run"]["mode"] == "ask" and s["run"]["max_tool_calls"] == 40


def test_add_data_not_committed(tmp_path):
    p = Project.init(tmp_path / "proj")
    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n")
    dest = p.add_data(csv)
    assert dest == p.root / "data/raw/d.csv" and dest.exists()
    p.checkpoint("data")
    assert "data/raw" not in git(p.root, "ls-files")


def test_ledger_roundtrip(tmp_path):
    p = Project.init(tmp_path)
    p.ledger_append({"a": 1})
    rec = p.ledger()[0]
    assert rec["a"] == 1 and "ts" in rec
