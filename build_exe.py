"""Utility script to package the VatiVision Pro application into a Windows executable.

This helper wraps ``PyInstaller`` with sensible defaults for the repository.
The default behaviour is to create a single-file executable from ``main.py``
using the repository's icon if it is available.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).parent.resolve()
DEFAULT_SCRIPT = REPO_ROOT / "main.py"
DEFAULT_ICON = REPO_ROOT / "program_logo.png"


class PyInstallerNotFoundError(RuntimeError):
    """Raised when PyInstaller cannot be located on the system PATH."""


class ScriptNotFoundError(FileNotFoundError):
    """Raised when the target script cannot be located."""


def _build_arguments(script: Path, *, icon: Path | None, name: str | None) -> List[str]:
    """Compose the argument list that will be passed to PyInstaller."""

    args = ["--noconfirm", "--onefile", "--clean"]

    if icon and icon.exists():
        args.extend(["--icon", str(icon)])

    if name:
        args.extend(["--name", name])

    args.append(str(script))
    return args


def _ensure_pyinstaller() -> str:
    """Return the absolute path to the ``pyinstaller`` executable.

    The function checks whether PyInstaller is available on the PATH.
    If PyInstaller is missing we raise an explicit error instead of
    failing later with a less helpful message.
    """

    executable = shutil.which("pyinstaller")
    if not executable:
        raise PyInstallerNotFoundError(
            "PyInstaller is not installed. Install it via 'pip install pyinstaller' "
            "before running this script."
        )
    return executable


def run_pyinstaller(script: Path, *, icon: Path | None, name: str | None) -> None:
    """Execute PyInstaller with the desired options."""

    pyinstaller_executable = _ensure_pyinstaller()
    args = [pyinstaller_executable, *_build_arguments(script, icon=icon, name=name)]

    process = subprocess.run(args, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            "PyInstaller exited with a non-zero status. Consult the log above "
            "for details."
        )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=DEFAULT_SCRIPT,
        help="Entry-point script to package (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the generated executable (defaults to the script name)",
    )
    parser.add_argument(
        "--icon",
        type=Path,
        default=DEFAULT_ICON if DEFAULT_ICON.exists() else None,
        help="Optional icon file to use for the executable.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    script_path = args.script if isinstance(args.script, Path) else Path(args.script)
    if not script_path.exists():
        raise ScriptNotFoundError(f"Target script '{script_path}' does not exist.")

    icon_path = args.icon if isinstance(args.icon, Path) else Path(args.icon)
    if args.icon is not None and not icon_path.exists():
        raise FileNotFoundError(f"Icon file '{icon_path}' does not exist.")

    run_pyinstaller(script_path, icon=icon_path if args.icon else None, name=args.name)


if __name__ == "__main__":
    main()
