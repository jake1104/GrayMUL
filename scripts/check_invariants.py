import argparse
import json
import re
import sys
from pathlib import Path

COMMENT_LINE = re.compile(r"^\s*//")


def check_no_docs(root):
    target = root / "docs"
    if target.exists():
        return False, "docs/ exists at %s" % target
    return True, ""


def check_no_colon_logs(root):
    offenders = []
    logs_root = root / "logs"
    if logs_root.exists():
        for path in sorted(logs_root.rglob("*")):
            if ":" in path.name:
                offenders.append(str(path))
    if offenders:
        return False, "colon names: %s" % offenders
    return True, ""


def check_no_source_comments(root):
    offenders = []
    crates_root = root / "crates"
    for path in sorted(crates_root.rglob("*.rs")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if COMMENT_LINE.match(line):
                offenders.append("%s:%d" % (path, number))
    if offenders:
        return False, "comment lines: %s" % offenders[:10]
    return True, ""


def check_pins_time_bands_only(root):
    pins_path = root / "bench" / "pins.json"
    try:
        text = pins_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, "cannot read %s: %s" % (pins_path, exc)
    for token in ('"elapsed_ms"', '"ref_ms"'):
        if token in text:
            return False, "time pin token %s present" % token
    try:
        pins = json.loads(text)
    except ValueError as exc:
        return False, "invalid JSON: %s" % exc
    cells = pins.get("cells")
    if not cells:
        return False, "no cells"
    for cell in cells:
        band = (cell.get("band_ms") or {}).get("upper")
        if not isinstance(band, (int, float)):
            return False, "cell %s lacks numeric band_ms.upper" % cell.get("name")
        expected = cell.get("expected") or {}
        for key in ("elapsed_ms", "ref_ms"):
            if key in expected:
                return False, "cell %s pins %s" % (cell.get("name"), key)
    return True, ""


def check_repo(root):
    root = Path(root)
    checks = [
        ("no-docs-touch", check_no_docs),
        ("no-colon-logs", check_no_colon_logs),
        ("no-source-comments", check_no_source_comments),
        ("pins-time-bands-only", check_pins_time_bands_only),
    ]
    failures = 0
    for name, func in checks:
        ok, detail = func(root)
        status = "PASS" if ok else "FAIL"
        extra = " (%s)" % detail if detail else ""
        print("[%s] %s%s" % (status, name, extra))
        if not ok:
            failures += 1
    if failures:
        print("RESULT: FAIL %d of %d invariants failed" % (failures, len(checks)))
        return 1
    print("RESULT: all %d invariants hold" % len(checks))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check repo invariants.")
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
