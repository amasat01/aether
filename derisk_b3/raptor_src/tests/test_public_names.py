# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The family's public-name coverage check (raptor.conformance.public_names) on a throwaway package."""
import subprocess

from raptor.conformance.public_names import main


def _tree(tmp_path, allow="", tested=("shown",)):
    pkg = tmp_path / "src" / "pkg"
    (pkg / "door").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from . import door\nfrom . import _hidden\n__all__ = ['shown', 'quiet']\n")
    (pkg / "door" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("\n".join(tested) + "\n")
    (tmp_path / "allow.txt").write_text(allow)
    return ["--src", str(tmp_path / "src"), "--tests", str(tmp_path / "tests"), "--allow", str(tmp_path / "allow.txt"),
            "--module", "pkg"]


def test_an_untested_name_and_an_untested_door_are_findings(tmp_path, capsys):
    assert main(_tree(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "no test references pkg.quiet" in out and "no test references pkg.door" in out
    assert "shown" not in out.split("RED:", 1)[1] and "_hidden" not in out


def test_allowlisted_names_pass_and_the_list_may_only_shrink(tmp_path, capsys):
    args = _tree(tmp_path, allow="pkg.quiet  # why\npkg.door  # why\n")
    assert main(args) == 0
    (tmp_path / "tests" / "test_x.py").write_text("shown quiet\n")
    assert main(args) == 1
    assert "pkg.quiet is referenced by a test now" in capsys.readouterr().out


def test_a_line_added_since_the_base_ref_is_a_finding(tmp_path, capsys):
    args = _tree(tmp_path, allow="pkg.quiet  # why\n")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "allow.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    (tmp_path / "allow.txt").write_text("pkg.quiet  # why\npkg.door  # why\n")
    assert main([*args, "--base", "HEAD"]) == 1
    assert "pkg.door was added" in capsys.readouterr().out


def test_a_starred_all_takes_the_items_of_its_module_level_literals(tmp_path, capsys):
    args = _tree(tmp_path)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text(
        "_NAMES = ('shown',)\n_DOORS = {'quiet': ('mod', 'quiet')}\n__all__ = ['door', *_NAMES, *_DOORS]\n")
    assert main(args) == 1
    out = capsys.readouterr().out
    assert "public names: 3" in out and "no test references pkg.quiet" in out
