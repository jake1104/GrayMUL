import argparse
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_HELP_FLAGS = [
    "--algorithm",
    "--timings",
    "--json",
    "--threads",
    "--timeout-seconds",
    "--max-memory-mb",
    "--max-cpu-seconds",
    "--seed",
    "--perf-mode",
    "--progress",
]

EXPECTED_PYTHON_CALLABLES = [
    "multiply",
    "multiply_report",
    "multiply_batch",
    "multiply_with_progress",
    "stage_timings",
    "version",
]

EXPECTED_ERROR_VARIANTS = ["Input", "ResourceLimit", "Cancelled", "Internal"]


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def check_repo(root):
    root = Path(root)
    cli_bin = root / "target" / "release" / "graymul.exe"
    error_path = root / "crates" / "graymul-core" / "src" / "error.rs"

    results = []

    def record(name, location, ok, detail=""):
        results.append({"name": name, "location": location, "ok": ok, "detail": detail})

    if not cli_bin.exists():
        print("MISMATCH: release CLI missing: %s" % cli_bin)
        return 1
    try:
        top = subprocess.run(
            [str(cli_bin), "--help"], capture_output=True, text=True, timeout=60
        )
        batch = subprocess.run(
            [str(cli_bin), "batch", "--help"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print("MISMATCH: cannot run CLI help: %s" % exc)
        return 1
    top_help = top.stdout + top.stderr
    batch_help = batch.stdout + batch.stderr
    for flag in EXPECTED_HELP_FLAGS:
        record(
            "CLI carries %s" % flag,
            str(cli_bin),
            flag in top_help,
            "%s missing from --help" % flag if flag not in top_help else "",
        )
    record(
        "CLI carries batch subcommand",
        str(cli_bin),
        "batch" in top_help,
        "batch missing from --help" if "batch" not in top_help else "",
    )
    record(
        "CLI batch carries --continue-on-error",
        str(cli_bin),
        "--continue-on-error" in batch_help,
        "--continue-on-error missing from batch --help"
        if "--continue-on-error" not in batch_help
        else "",
    )

    try:
        import graymul
    except ImportError as exc:
        print("MISMATCH: cannot import graymul: %s" % exc)
        return 1
    missing = [name for name in EXPECTED_PYTHON_CALLABLES if not callable(getattr(graymul, name, None))]
    error_ok = isinstance(graymul.GrayMULError, type) and issubclass(graymul.GrayMULError, ValueError)
    python_detail = ""
    if missing:
        python_detail += "missing=%s " % missing
    if not error_ok:
        python_detail += "GrayMULError is not a ValueError subclass"
    record(
        "python exports callable with GrayMULError",
        "graymul.__init__",
        not missing and error_ok,
        python_detail.strip(),
    )

    try:
        error_text = read_text(error_path)
    except OSError as exc:
        print("MISMATCH: cannot read %s: %s" % (error_path, exc))
        return 1
    missing_variants = [name for name in EXPECTED_ERROR_VARIANTS if name not in error_text]
    record(
        "core error categories present",
        str(error_path),
        not missing_variants,
        "missing=%s" % missing_variants if missing_variants else "",
    )

    failures = [entry for entry in results if not entry["ok"]]
    for entry in results:
        status = "OK" if entry["ok"] else "MISMATCH"
        extra = " (%s)" % entry["detail"] if entry["detail"] else ""
        print("[%s] %s (%s)%s" % (status, entry["name"], entry["location"], extra))

    total = len(results)
    passed = total - len(failures)
    if failures:
        print("RESULT: MISMATCH %d of %d checks failed" % (len(failures), total))
        return 1
    print("RESULT: all %d of %d checks pass" % (passed, total))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check CLI/Python/core API surface.")
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root override (default: parent of scripts dir).",
    )
    args = parser.parse_args(argv)
    if args.root is not None:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parent.parent
    return check_repo(root)


if __name__ == "__main__":
    sys.exit(main())
