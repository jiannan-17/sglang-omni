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
        'fun-cosyvoice3 = ["onnx>=1.18.0", "setuptools<80"]\n'
    )
    return path


@pytest.mark.parametrize(
    ("extra", "package", "installed_version", "expected"),
    [
        ("minicpm-o", "onnx", None, ["onnx>=1.18.0"]),
        ("minicpm-o", "onnx", "1.17.0", ["onnx>=1.18.0"]),
        ("minicpm-o", "onnx", "1.18.0", []),
        ("fun-cosyvoice3", "setuptools", None, ["setuptools<80"]),
        ("fun-cosyvoice3", "setuptools", "79.0.1", []),
        ("fun-cosyvoice3", "setuptools", "80.0.0", ["setuptools<80"]),
    ],
)
def test_selected_extra_checks_dependency_version_constraints(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: str,
    package: str,
    installed_version: str | None,
    expected: list[str],
) -> None:
    versions = {"torch": "2.13.0", "einops": "0.8.1", "onnx": "1.18.0"}
    versions[package] = installed_version

    def version(name: str) -> str:
        installed = versions[name]
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        else:
            return installed

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    assert dependencies.missing_requirements(project) == []
    assert dependencies.missing_requirements(project, (extra,)) == expected


def test_unknown_extra_fails_instead_of_silently_omitting_dependencies(
    project: Path,
) -> None:
    with pytest.raises(KeyError, match="minicpm-typo"):
        dependencies.missing_requirements(project, ("minicpm-typo",))


def test_source_paths_survive_overwritten_pythonpath(tmp_path: Path) -> None:
    virtualenv = tmp_path / "ci home" / "omni"
    venv.EnvBuilder(with_pip=False).create(virtualenv)
    source = virtualenv / "src" / "CosyVoice"
    packages = [
        source / "cosyvoice",
        source / "third_party" / "Matcha-TTS" / "matcha",
    ]
    for package in packages:
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}

    # note (Jiannan Li): CI covers checkout; this test exercises Python's .pth loading.
    subprocess.run(
        [
            "bash",
            "-c",
            'git() { :; }; source "$@"',
            "bash",
            str(SCRIPT.with_name("prepare_cosyvoice_sources.sh")),
            str(virtualenv),
            "unused-repository",
            "unused-revision",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            str(virtualenv / "bin/python"),
            "-c",
            "import json, cosyvoice, matcha; "
            "print(json.dumps([cosyvoice.__file__, matcha.__file__]))",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert [Path(path).resolve() for path in json.loads(result.stdout)] == [
        (package / "__init__.py").resolve() for package in packages
    ]
