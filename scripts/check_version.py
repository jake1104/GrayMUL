import argparse
import json
import re
import sys
from pathlib import Path


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def section_body(text, section):
    pattern = re.compile(
        r"^\[" + re.escape(section) + r"\](.*?)(?=^\[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def extract_section_version(text, section):
    body = section_body(text, section)
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', body, re.MULTILINE)
    return match.group(1).strip() if match else None


def check_repo(root):
    root = Path(root)
    cargo_path = root / "Cargo.toml"
    pyproject_path = root / "pyproject.toml"
    test_path = root / "python" / "tests" / "test_mul.py"
    pins_path = root / "bench" / "pins.json"
    crate_names = ["graymul-core", "graymul-cli", "graymul-py", "graymul-bench", "graymul-wasm"]

    results = []

    def record(name, location, found, expected):
        ok = found is not None and found == expected
        results.append(
            {"name": name, "location": location, "found": found, "expected": expected, "ok": ok}
        )

    try:
        cargo_text = read_text(cargo_path)
    except OSError as exc:
        print("MISMATCH: cannot read %s: %s" % (cargo_path, exc))
        return 1
    cargo_version = extract_section_version(cargo_text, "workspace.package")
    if cargo_version is None:
        print("MISMATCH: no version under [workspace.package] in %s" % cargo_path)
        return 1
    print("canonical Cargo [workspace.package] version: %s" % cargo_version)

    try:
        pyproject_text = read_text(pyproject_path)
    except OSError as exc:
        print("MISMATCH: cannot read %s: %s" % (pyproject_path, exc))
        return 1
    record(
        "pyproject.toml [project] version",
        str(pyproject_path),
        extract_section_version(pyproject_text, "project"),
        cargo_version,
    )

    for crate in crate_names:
        crate_path = root / "crates" / crate / "Cargo.toml"
        try:
            crate_text = read_text(crate_path)
        except OSError as exc:
            print("MISMATCH: cannot read %s: %s" % (crate_path, exc))
            return 1
        uses_workspace = re.search(r"^\s*version\.workspace\s*=\s*true", crate_text, re.MULTILINE) is not None
        record(
            "%s tracks workspace version" % crate,
            str(crate_path),
            "workspace" if uses_workspace else None,
            "workspace",
        )

    try:
        test_text = read_text(test_path)
    except OSError as exc:
        print("MISMATCH: cannot read %s: %s" % (test_path, exc))
        return 1
    assertion = re.search(
        r"graymul\.version\(\)\s*==\s*[\"'](\d+\.\d+\.\d+)[\"']", test_text
    )
    record(
        "test_mul.py version assertion",
        "%s (graymul.version() ==)" % test_path,
        assertion.group(1) if assertion else None,
        cargo_version,
    )

    try:
        pins_text = read_text(pins_path)
    except OSError as exc:
        print("MISMATCH: cannot read %s: %s" % (pins_path, exc))
        return 1
    try:
        pins_version = json.loads(pins_text).get("version")
    except ValueError as exc:
        print("MISMATCH: %s is not valid JSON: %s" % (pins_path, exc))
        return 1
    record("bench/pins.json version", str(pins_path), pins_version, cargo_version)

    failures = [entry for entry in results if not entry["ok"]]
    for entry in results:
        status = "OK" if entry["ok"] else "MISMATCH"
        found = entry["found"] if entry["found"] is not None else "<missing>"
        print(
            "[%s] %s: found=%s expected=%s (%s)"
            % (status, entry["name"], found, entry["expected"], entry["location"])
        )

    if failures:
        print("RESULT: MISMATCH %d of %d checks failed" % (len(failures), len(results)))
        return 1
    print("RESULT: all %d checks match %s" % (len(results), cargo_version))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check repo version drift.")
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
