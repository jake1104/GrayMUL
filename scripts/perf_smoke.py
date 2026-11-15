import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SUBPROCESS_TIMEOUT_SECONDS = 300
ALLOWED_RATIO = 2.0
GRANULARITY_FLOOR_MS = 50
RATIO_REPS = 3


try:
    sys.set_int_max_str_digits(20000)
except AttributeError:
    pass


def load_pins(root):
    pins_path = root / "bench" / "pins.json"
    with open(pins_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_forced(root):
    forced_path = root / "bench" / "forced.json"
    with open(forced_path, encoding="utf-8") as handle:
        return json.load(handle)


def median_of(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def run_capture(args):
    started = time.perf_counter()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "subprocess timeout", "wall_ms": (time.perf_counter() - started) * 1000.0}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def last_json_line(stdout):
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError:
        return None


def run_bench_cell(root, logs_dir, cell):
    bench_bin = str(root / "target" / "release" / "graymul-bench.exe")
    out_path = str(logs_dir / ("smoke-bench-%s.json" % cell["name"]))
    args = [bench_bin, "--tier", cell["name"], "--task-id", "perf-smoke", "--output", out_path]
    captured = run_capture(args)
    if not captured["ok"]:
        return {"verdict": "FAIL", "error": "bench exit %s" % captured.get("returncode"), "stderr": captured["stderr"][-500:]}
    try:
        with open(out_path, encoding="utf-8") as handle:
            archive = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"verdict": "FAIL", "error": "bench archive unreadable: %s" % exc}
    observations = archive.get("observations", [])
    if not observations or not all(item.get("verified") for item in observations):
        return {"verdict": "FAIL", "error": "bench observations unverified"}
    return {"verdict": "PASS", "observations": len(observations)}


def run_cli_cell(root, cell):
    cli_bin = str(root / "target" / "release" / "graymul.exe")
    expected = cell["expected"]
    captured = run_capture([cli_bin, cell["input_a"], cell["input_b"], "--json"])
    if not captured["ok"]:
        return {"verdict": "FAIL", "error": "cli exit %s" % captured.get("returncode"), "stderr": captured["stderr"][-500:]}
    payload = last_json_line(captured["stdout"])
    if payload is None:
        return {"verdict": "FAIL", "error": "cli stdout is not JSON"}
    problems = []
    target = int(cell["input_a"]) * int(cell["input_b"])
    try:
        product = int(payload.get("product"))
    except (TypeError, ValueError):
        product = None
    if product != target:
        problems.append("product mismatch")
    if payload.get("method") != expected["method"]:
        problems.append("method %s != %s" % (payload.get("method"), expected["method"]))
    if payload.get("limbs") != expected["limbs"]:
        problems.append("limbs %s != %s" % (payload.get("limbs"), expected["limbs"]))
    elapsed = payload.get("elapsed_ms")
    upper = cell["band_ms"]["upper"]
    band_ok = isinstance(elapsed, (int, float)) and elapsed <= upper
    if not band_ok:
        problems.append("elapsed %s outside band upper %s" % (elapsed, upper))
    return {
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
        "elapsed_ms": elapsed,
        "band_upper_ms": upper,
    }


def run_python_cell(cell):
    try:
        import graymul
    except ImportError as exc:
        return {"verdict": "SKIP", "error": "graymul not installed: %s" % exc}
    try:
        product = graymul.multiply(int(cell["input_a"]), int(cell["input_b"]))
    except Exception as exc:
        return {"verdict": "FAIL", "error": "%s: %s" % (type(exc).__name__, exc)}
    if product != int(cell["input_a"]) * int(cell["input_b"]):
        return {"verdict": "FAIL", "error": "product mismatch"}
    return {"verdict": "PASS", "product_digits": len(str(product))}


def run_cli_timed(root, cell, engine):
    cli_bin = str(root / "target" / "release" / "graymul.exe")
    target = int(cell["input_a"]) * int(cell["input_b"])
    samples = []
    for _ in range(RATIO_REPS):
        args = [cli_bin, cell["input_a"], cell["input_b"], "--json"]
        if engine is not None:
            args += ["--algorithm", engine]
        captured = run_capture(args)
        if not captured["ok"]:
            return {"verdict": "FAIL", "error": "cli exit %s" % captured.get("returncode")}
        payload = last_json_line(captured["stdout"])
        if payload is None:
            return {"verdict": "FAIL", "error": "cli stdout is not JSON"}
        try:
            product = int(payload.get("product"))
        except (TypeError, ValueError):
            product = None
        if product != target:
            return {"verdict": "FAIL", "error": "product mismatch"}
        elapsed = payload.get("elapsed_ms")
        if not isinstance(elapsed, (int, float)):
            return {"verdict": "FAIL", "error": "elapsed_ms missing"}
        samples.append(float(elapsed))
    return {"verdict": "PASS", "median_ms": median_of(samples)}


def run_ratio_cell(root, cell, forced):
    engine = forced.get("cells", {}).get(cell["name"])
    if engine is None:
        return {"verdict": "FAIL", "error": "no forced engine for %s" % cell["name"]}
    auto = run_cli_timed(root, cell, None)
    if auto["verdict"] != "PASS":
        return {"verdict": "FAIL", "error": "auto leg: %s" % auto.get("error")}
    forced_run = run_cli_timed(root, cell, engine)
    if forced_run["verdict"] != "PASS":
        return {"verdict": "FAIL", "error": "forced leg: %s" % forced_run.get("error")}
    auto_med = auto["median_ms"]
    forced_med = forced_run["median_ms"]
    limit = forced_med + max(GRANULARITY_FLOOR_MS, forced_med * (ALLOWED_RATIO - 1.0))
    verdict = "PASS" if auto_med <= limit else "FAIL"
    return {
        "verdict": verdict,
        "engine": engine,
        "auto_ms": auto_med,
        "forced_ms": forced_med,
        "limit_ms": limit,
    }


def run_cell(root, logs_dir, cell, forced):
    bench = run_bench_cell(root, logs_dir, cell)
    cli = run_cli_cell(root, cell)
    python_result = run_python_cell(cell)
    ratio = run_ratio_cell(root, cell, forced)
    verdict = "PASS" if all(part["verdict"] in ("PASS", "SKIP") for part in (bench, cli, python_result, ratio)) else "FAIL"
    print(
        "%s bench=%s cli=%s python=%s ratio=%s auto_ms=%s forced_ms=%s forced_engine=%s limit_ms=%s elapsed_ms=%s band_upper_ms=%s verdict=%s"
        % (
            cell["name"],
            bench["verdict"],
            cli["verdict"],
            python_result["verdict"],
            ratio["verdict"],
            ratio.get("auto_ms"),
            ratio.get("forced_ms"),
            ratio.get("engine"),
            ratio.get("limit_ms"),
            cli.get("elapsed_ms"),
            cell["band_ms"]["upper"],
            verdict,
        )
    )
    for part_name, part in (("bench", bench), ("cli", cli), ("python", python_result), ("ratio", ratio)):
        if part["verdict"] not in ("PASS", "SKIP"):
            print("  %s problem: %s" % (part_name, part.get("error") or part.get("problems")))
    sys.stdout.flush()
    return {"name": cell["name"], "bench": bench, "cli": cli, "python": python_result, "ratio": ratio, "verdict": verdict}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Perf smoke gate for GrayMUL release surface.")
    parser.add_argument("--root", default=None, help="Repo root override (default: parent of scripts dir).")
    parser.add_argument("--cells", default=None, help="Comma-separated cell subset, default runs all pins cells.")
    parser.add_argument("--logs-dir", default=None, help="Bench JSON output dir (default logs/perf-smoke).")
    args = parser.parse_args(argv)
    if args.root is not None:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parent.parent
    pins = load_pins(root)
    wanted = None
    if args.cells:
        wanted = set(part.strip() for part in args.cells.split(",") if part.strip())
    cells = pins["cells"]
    if wanted is not None:
        unknown = wanted - set(cell["name"] for cell in cells)
        if unknown:
            print("ERROR unknown cells: %s" % sorted(unknown))
            return 2
        cells = [cell for cell in cells if cell["name"] in wanted]
    if args.logs_dir is not None:
        logs_dir = Path(args.logs_dir)
    else:
        logs_dir = root / "logs" / "perf-smoke"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print("perf_smoke cells=%s" % [cell["name"] for cell in cells])
    sys.stdout.flush()
    forced = load_forced(root)
    results = [run_cell(root, logs_dir, cell, forced) for cell in cells]
    overall = all(item["verdict"] == "PASS" for item in results)
    print("OVERALL verdict=%s" % ("PASS" if overall else "FAIL"))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
