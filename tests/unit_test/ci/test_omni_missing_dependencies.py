# SPDX-License-Identifier: Apache-2.0
"""Dependency checks must cover the optional models selected by CI."""

import os
import subprocess
import sys
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
        'fun-cosyvoice3 = ["conformer==0.3.2", "setuptools<80"]\n'
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


@pytest.mark.parametrize(
    ("conformer_version", "setuptools_version", "expected"),
    [
        (None, "79.0.1", ["conformer==0.3.2"]),
        ("0.3.1", "79.0.1", ["conformer==0.3.2"]),
        ("0.3.2", "80.0.0", ["setuptools<80"]),
        ("0.3.2", "79.0.1", []),
    ],
)
def test_ci_extras_cli_checks_cosyvoice_dependencies(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    conformer_version: str | None,
    setuptools_version: str,
    expected: list[str],
) -> None:
    versions = {
        "torch": "2.13.0",
        "einops": "0.8.1",
        "onnx": "1.18.0",
        "conformer": conformer_version,
        "setuptools": setuptools_version,
    }

    def version(name: str) -> str:
        installed = versions[name]
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        else:
            return installed

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    arguments = [
        str(SCRIPT),
        "--extra",
        "minicpm-o",
        "--extra",
        "fun-cosyvoice3",
        str(project),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert dependencies.main() == 0
    assert capsys.readouterr().out.split() == expected

    monkeypatch.setattr(sys, "argv", [*arguments, "--check"])
    assert dependencies.main() == int(bool(expected))
    output = capsys.readouterr().out
    for requirement in expected:
        assert requirement in output


@pytest.mark.parametrize("missing_package", [None, "cosyvoice", "matcha"])
def test_import_probe_rejects_missing_cosyvoice_sources(
    tmp_path: Path, missing_package: str | None
) -> None:
    modules = {
        "av.py": "",
        "llama_cpp.py": "",
        "torch.py": "",
        "transformers.py": "",
        "sglang.py": "",
        "zhon/hanzi.py": "",
        "whisper/normalizers.py": "EnglishTextNormalizer = None\n",
        "sglang_omni/models/qwen3_tts/compat.py": (
            "def apply_qwen_tts_transformers_compatibility_patches(): pass\n"
        ),
        "qwen_tts.py": "Qwen3TTSModel = Qwen3TTSTokenizer = None\n",
        "dac.py": "",
        "neucodec.py": "NeuCodec = None\n",
        "cosyvoice/cli/cosyvoice.py": "CosyVoice3 = None\n",
        "matcha/models/components/flow_matching.py": "BASECFM = None\n",
    }
    for relative_path, content in modules.items():
        if relative_path.split("/")[0] == missing_package:
            continue
        else:
            target = tmp_path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    venv.EnvBuilder(with_pip=False).create(tmp_path / "omni")
    executable = tmp_path / "omni/bin/python"
    sox = executable.with_name("sox")
    sox.write_text("#!/bin/sh\nexit 0\n")
    sox.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("validate_omni_venv_imports.sh")), "omni"],
        cwd=tmp_path,
        env={
            **os.environ,
            "OMNI_CI_HOME": str(tmp_path),
            "PYTHONPATH": str(tmp_path),
            "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if missing_package is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert f"No module named '{missing_package}'" in result.stderr


def test_ci_dependency_fingerprint_tracks_source_preparation(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\ndependencies = []\n")
    prepare = tmp_path / "prepare_omni_venv.sh"
    prepare.write_text("COSYVOICE_COMMIT=old\n")
    fingerprint = tmp_path / "omni_ci_deps_hash.sh"
    fingerprint.write_text(SCRIPT.with_name("omni_ci_deps_hash.sh").read_text())
    hashes = []
    for content in ("COSYVOICE_COMMIT=old\n", "COSYVOICE_COMMIT=new\n"):
        prepare.write_text(content)
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; omni_ci_deps_hash', "bash", str(fingerprint)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        hashes.append(result.stdout.strip())
    assert hashes[0] != hashes[1]
