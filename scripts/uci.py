from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ARCHITECTURES = ("gadus", "melano")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def uci_executable(architecture: str) -> Path:
	suffix = ".exe" if os.name == "nt" else ""
	path = PROJECT_ROOT / "build" / architecture / f"uci{suffix}"
	if not path.is_file():
		raise FileNotFoundError(
			f"{architecture} UCI engine is missing: {path}. Run the build script first."
		)
	return path


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Launch an architecture-specific Gadidae UCI engine"
	)
	parser.add_argument("--arch", choices=ARCHITECTURES, required=True)
	parser.add_argument("--model", required=True)
	args, remainder = parser.parse_known_args(argv)

	model = Path(args.model)
	if not model.is_absolute():
		model = PROJECT_ROOT / model
	if not model.is_file():
		raise FileNotFoundError(f"model not found: {model}")

	command = [
		str(uci_executable(args.arch)),
		"--model",
		str(model),
		*remainder,
	]
	if os.name != "nt":
		os.execv(command[0], command)
	return subprocess.call(command)


if __name__ == "__main__":
	raise SystemExit(main())
