# SPDX-License-Identifier: Apache-2.0
"""Dependency checks must cover the optional models selected by CI."""

import json
import os
import subprocess
import venv
from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/omni_missing_dependencies.py"
)
SPEC = spec_from_file_location("omni_missing_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dependencies = module_from_spec(SPEC)
SPEC.loader.exec_module(dependencies)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\ndependencies = ["torch==2.13.0"]\n'
        "[project.optional-dependencies]\n"
        'minicpm-o = ["einops>=0.8.1", "onnx>=1.18.0"]\n'
    )
    return path


@pytest.mark.parametrize("onnx_version", [None, "1.17.0", "1.18.0"])
def test_minicpm_extra_checks_missing_and_outdated_dependencies(
    project: Path, monkeypatch: pytest.MonkeyPatch, onnx_version: str | None
) -> None:
    versions = {"torch": "2.13.0", "einops": "0.8.1", "onnx": onnx_version}

    def version(name: str) -> str:
        installed = versions[name]
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    assert dependencies.missing_requirements(project) == []
    assert dependencies.missing_requirements(project, ("minicpm-o",)) == (
        [] if onnx_version == "1.18.0" else ["onnx>=1.18.0"]
    )


def test_unknown_extra_fails_instead_of_silently_omitting_dependencies(
    project: Path,
) -> None:
    with pytest.raises(KeyError, match="minicpm-typo"):
        dependencies.missing_requirements(project, ("minicpm-typo",))


def run_git(repository: Path, arguments: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def source_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    for key, value in {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_AUTHOR_NAME": "CI Test",
        "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI Test",
        "GIT_COMMITTER_EMAIL": "ci@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)

    for package in ("matcha", "cosyvoice"):
        repository = tmp_path / f"{package}-repository"
        (repository / package).mkdir(parents=True)
        (repository / package / "__init__.py").write_text("REVISION = 'pinned'\n")
        run_git(repository, ["init", "--quiet"])
        run_git(repository, ["add", "."])
        run_git(repository, ["commit", "--quiet", "-m", "Initial source"])

    cosyvoice = tmp_path / "cosyvoice-repository"
    matcha = tmp_path / "matcha-repository"
    run_git(
        cosyvoice,
        ["submodule", "add", matcha.as_uri(), "third_party/Matcha-TTS"],
    )
    run_git(cosyvoice, ["commit", "--quiet", "-am", "Pin Matcha source"])
    revision = run_git(cosyvoice, ["rev-parse", "HEAD"])
    for repository, package in ((cosyvoice, "cosyvoice"), (matcha, "matcha")):
        (repository / package / "__init__.py").write_text("REVISION = 'newer'\n")
        run_git(repository, ["commit", "--quiet", "-am", "Advance source branch"])
    return cosyvoice, revision


@pytest.mark.parametrize("valid_revision", [True, False])
def test_source_setup_with_overwritten_pythonpath(
    tmp_path: Path, source_repository: tuple[Path, str], valid_revision: bool
) -> None:
    repository, revision = source_repository
    virtualenv = tmp_path / "ci home" / "omni"
    venv.EnvBuilder(with_pip=False).create(virtualenv)
    project = tmp_path / "project"
    project.mkdir()
    environment = {**os.environ, "PYTHONPATH": str(project)}
    probe = [
        str(virtualenv / "bin/python"),
        "-c",
        "import json, cosyvoice, matcha; "
        "print(json.dumps([[package.REVISION, package.__file__] "
        "for package in (cosyvoice, matcha)]))",
    ]
    before = subprocess.run(
        probe, cwd=project, env=environment, capture_output=True, text=True
    )
    assert before.returncode != 0
    assert "No module named 'cosyvoice'" in before.stderr

    setup = subprocess.run(
        [
            "bash",
            str(SCRIPT.with_name("prepare_cosyvoice_sources.sh")),
            str(virtualenv),
            repository.as_uri(),
            revision if valid_revision else "0" * 40,
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
    )
    after = subprocess.run(
        probe, cwd=project, env=environment, capture_output=True, text=True
    )
    if valid_revision:
        assert setup.returncode == 0, setup.stderr
        assert after.returncode == 0, after.stderr
        for installed_revision, source_path in json.loads(after.stdout):
            assert installed_revision == "pinned"
            assert Path(source_path).is_relative_to(virtualenv.resolve())
    else:
        assert setup.returncode != 0
        assert after.returncode != 0


@pytest.mark.parametrize(
    "changed_script", ["prepare_omni_venv.sh", "prepare_cosyvoice_sources.sh"]
)
def test_ci_dependency_fingerprint_tracks_source_preparation(
    tmp_path: Path, changed_script: str
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\ndependencies = []\n")
    for name in ("prepare_omni_venv.sh", "prepare_cosyvoice_sources.sh"):
        (tmp_path / name).write_text(SCRIPT.with_name(name).read_text())
    fingerprint = tmp_path / "omni_ci_deps_hash.sh"
    fingerprint.write_text(SCRIPT.with_name("omni_ci_deps_hash.sh").read_text())
    hashes = []
    for appended_content in ("", "\n# Updated source preparation\n"):
        with (tmp_path / changed_script).open("a") as script:
            script.write(appended_content)
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; omni_ci_deps_hash', "bash", str(fingerprint)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        hashes.append(result.stdout.strip())
    assert hashes[0] != hashes[1]
