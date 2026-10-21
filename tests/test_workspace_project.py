import os
import subprocess

import pytest

from beyondml.workspace.project import PathError, Project, ProjectError


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


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
    with pytest.raises(PathError):
        p.path(".git/hooks/pre-commit", write=True)
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


def test_restore_bad_sha_raises(tmp_path):
    p = Project.init(tmp_path)
    p.write("a.txt", "x")
    with pytest.raises(ProjectError):
        p.restore("garbage", ["a.txt"])
    assert p.read("a.txt") == "x"


def test_restore_refuses_root_and_git(tmp_path):
    p = Project.init(tmp_path)
    sha = p.checkpoint("c")
    for bad in [".", ".git", ".beyondml/settings.toml"]:
        with pytest.raises(PathError):
            p.restore(sha, [bad])


def test_settings_not_aliased(tmp_path):
    p = Project.init(tmp_path)
    p.settings()["run"]["mode"] = "mutated"
    assert p.settings()["run"]["mode"] == "auto"


def test_lock_shared_across_instances(tmp_path):
    p = Project.init(tmp_path)
    assert Project.open(tmp_path)._lock is Project.open(tmp_path)._lock is p._lock


def test_init_refuses_non_empty_folder(tmp_path):
    (tmp_path / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ProjectError, match="not empty"):
        Project.init(tmp_path)
    assert not (tmp_path / ".beyondml").exists()


def test_init_refuses_existing_git_repo(tmp_path):
    git(tmp_path, "init")
    with pytest.raises(ProjectError):
        Project.init(tmp_path)
    assert not (tmp_path / ".beyondml").exists()


def test_gitignore_ignores_pycache(tmp_path):
    assert "__pycache__/" in Project.init(tmp_path).read(".gitignore")


def test_commits_ignore_gpgsign_and_hooks(tmp_path, monkeypatch):
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = no-such-gpg-binary\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    root = tmp_path / "p"
    p = Project.init(root)  # the first commit must not try to sign
    (root / ".git/hooks/pre-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    p.write("a.txt", "x")
    before = git(root, "rev-parse", "HEAD").strip()
    assert p.checkpoint("x") != before
