#!/usr/bin/env python3
"""Durable, host-local queue for the 96 calendar-token EIIL3 downstream cases.

The 72 earlier A0 pilots are left on their original hosts. The 24 horizon-192
pilots have a fixed host assignment. Four screen sessions (one per GPU) run this
queue independently; source and output checks make a restart safe.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BACKBONES = ("patchtst", "moderntcn", "itransformer")
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather",
            "Electricity", "ExchangeRate", "Solar")
HORIZONS = (96, 192, 336, 720)
OLD_HORIZONS = (96, 336, 720)
OLD_ROOTS = (
    "fpem_periodic_token_a0_global_20261118",
    "fpem_periodic_token_a0_exchange_global_20261118",
    "fpem_periodic_token_a0_solar_global_20261118",
)
H192_ROOTS = {
    "general": "fpem_periodic_token_a0_global_h192_20261118",
    "ExchangeRate": "fpem_periodic_token_a0_exchange_global_h192_20261118",
    "Solar": "fpem_periodic_token_a0_solar_global_h192_20261118",
}
H192_DATASET_GPU = {
    "server1": {"Solar": 0, "ETTm1": 1, "ETTh1": 2, "ExchangeRate": 3},
    "server2": {"Electricity": 0, "Weather": 1, "ETTm2": 2, "ETTh2": 3},
}
EXPECTED_OLD_COUNT = {"server1": 37, "server2": 35}
HEAVY_DATASETS = {"Electricity", "Weather", "Solar"}
RAM_SLOT_PATHS = tuple(
    Path(f"/tmp/fpem_periodic_downstream_ram_{number}.lock")
    for number in range(4)
)
# Keep the original path as slot 0: queues started with the previous scheduler
# still hold this lock, so a rolling restart never admits a third heavy case.
HEAVY_SLOT_PATHS = (
    Path("/tmp/fpem_periodic_downstream_heavy.lock"),
    Path("/tmp/fpem_periodic_downstream_heavy_1.lock"),
)
ADMISSION_LOCK = Path("/tmp/fpem_periodic_downstream_admission.lock")
START_SPACING_SECONDS = 60
# A Solar case has been observed at about 94 GiB RSS. A second heavy launch
# needs room for that footprint *and* roughly 96 GiB of host safety margin.
HEAVY_MIN_AVAILABLE_GIB = 190
LIGHT_MIN_AVAILABLE_GIB = 64
REQUIRED_VERSIONS = {"A0", "F03", "R3", "R3(trick)", "F03+R3", "F03+R3(reverse)"}
PILOT_RE = re.compile(
    r"^pilot_(patchtst|moderntcn|itransformer)_"
    r"(ETTh1|ETTh2|ETTm1|ETTm2|Weather|Electricity|ExchangeRate|Solar)_"
    r"(96|192|336|720)$"
)


@dataclass(frozen=True)
class Case:
    backbone: str
    dataset: str
    horizon: int
    source: Path

    @property
    def key(self) -> str:
        return f"{self.backbone}/{self.dataset}/pred_{self.horizon}/seed_2021"

    @property
    def slug(self) -> str:
        return f"{self.backbone}_{self.dataset}_{self.horizon}"


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "event": event, **fields}, sort_keys=True), flush=True)


def old_cases(repo: Path, host: str) -> list[Case]:
    found: dict[tuple[str, str, int], Case] = {}
    for name in OLD_ROOTS:
        root = repo / "results" / name
        if not root.is_dir():
            continue
        for path in root.glob("pilot_*"):
            if not path.is_dir():
                continue
            match = PILOT_RE.fullmatch(path.name)
            if match is None:
                raise RuntimeError(f"unrecognized A0 pilot directory: {path}")
            backbone, dataset, length = match.groups()
            horizon = int(length)
            if horizon not in OLD_HORIZONS:
                raise RuntimeError(f"unexpected horizon in old A0 root: {path}")
            key = (backbone, dataset, horizon)
            if key in found:
                raise RuntimeError(f"duplicate A0 pilot: {found[key].source} and {path}")
            found[key] = Case(backbone, dataset, horizon, path)
    if len(found) != EXPECTED_OLD_COUNT[host]:
        raise RuntimeError(
            f"{host} has {len(found)} old A0 pilots; expected "
            f"{EXPECTED_OLD_COUNT[host]}. Refusing a partial matrix."
        )
    return sorted(found.values(), key=lambda c: (DATASETS.index(c.dataset),
                                                 BACKBONES.index(c.backbone),
                                                 OLD_HORIZONS.index(c.horizon)))


def horizon_192_cases(repo: Path, host: str) -> list[tuple[int, Case]]:
    cases = []
    for dataset, gpu in H192_DATASET_GPU[host].items():
        root_name = H192_ROOTS.get(dataset, H192_ROOTS["general"])
        for backbone in BACKBONES:
            source = repo / "results" / root_name / f"pilot_{backbone}_{dataset}_192"
            cases.append((gpu, Case(backbone, dataset, 192, source)))
    return cases


def make_plan(repo: Path, host: str) -> dict[int, list[Case]]:
    if host not in H192_DATASET_GPU:
        raise ValueError(f"unknown host {host}")
    plan: dict[int, list[Case]] = {gpu: [] for gpu in range(4)}
    for index, case in enumerate(old_cases(repo, host)):
        plan[index % 4].append(case)
    for gpu, case in horizon_192_cases(repo, host):
        plan[gpu].append(case)
    keys = [case.key for group in plan.values() for case in group]
    if len(keys) != EXPECTED_OLD_COUNT[host] + 12 or len(keys) != len(set(keys)):
        raise RuntimeError("downstream plan has missing or duplicate cases")
    return plan


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_ready(case: Case, *, verify_weights: bool = True) -> bool:
    source = case.source
    required = (source / "complete", source / "protocol.json",
                source / "comparison.json", source / "token" / "best_val.pt")
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    protocol = read_json(source / "protocol.json")
    expected = {"backbone": case.backbone, "dataset": case.dataset,
                "pred_len": case.horizon, "seed": 2021,
                "max_train_batches": 0, "max_eval_batches": 0}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"A0 source {source} has {key}={protocol.get(key)!r}, "
                               f"expected {value!r}")
    comparison = read_json(source / "comparison.json")
    if not isinstance(comparison.get("token"), dict):
        raise RuntimeError(f"A0 comparison lacks token result: {source}")
    if verify_weights:
        import torch
        payload = torch.load(source / "token" / "best_val.pt", map_location="cpu",
                             weights_only=False)
        if payload.get("arm") != "token" or not isinstance(payload.get("state_dict"), dict):
            raise RuntimeError(f"A0 token checkpoint malformed: {source}")
        keys = payload["state_dict"].keys()
        if case.backbone == "itransformer":
            # iTransformer inserts calendar variables into its existing global
            # attention; it does not add an exogenous parameter namespace.
            if ("inverted_embedding.value_embedding.weight" not in keys
                    or protocol.get("calendar_features", 0) <= 0):
                raise RuntimeError(f"iTransformer token source is malformed: {source}")
        elif not any("global_exog_attention" in key for key in keys):
            raise RuntimeError(f"A0 checkpoint lacks global token attention: {source}")
    return True


def complete_output(output: Path, case: Case, a0_sha: str) -> bool:
    marker = output / "downstream_complete"
    metrics_path = output / "metrics.json"
    if not marker.is_file() or not metrics_path.is_file():
        return False
    metrics = read_json(metrics_path)
    if metrics.get("status") != "downstream_complete":
        raise RuntimeError(f"completion marker but wrong metrics status: {output}")
    if metrics.get("A0_sha256") != a0_sha:
        raise RuntimeError(f"completion marker but A0 hash differs: {output}")
    if metrics.get("selection_split") != "test":
        raise RuntimeError(f"completion marker but selection split differs: {output}")
    if metrics.get("case_key", metrics.get("case")) != case.key:
        raise RuntimeError(f"completion marker but case key differs: {output}")
    versions = metrics.get("versions")
    if not isinstance(versions, dict) or not REQUIRED_VERSIONS.issubset(versions):
        raise RuntimeError(f"completion marker but versions are missing: {output}")
    return True


def h192_screen_active(gpu: int) -> bool:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    return bool(re.search(rf"\.fpem_periodic_h192_g{gpu}\b", result.stdout))


def wait_for_gpu_a0(gpu: int, poll_seconds: float) -> None:
    reported = False
    while h192_screen_active(gpu):
        if not reported:
            emit("WAIT_A0_SCREEN", gpu=gpu)
            reported = True
        time.sleep(poll_seconds)
    if reported:
        emit("A0_SCREEN_EXITED", gpu=gpu)


def wait_for_source(case: Case, gpu: int, poll_seconds: float) -> None:
    reported = False
    while not source_ready(case, verify_weights=False):
        if case.horizon != 192:
            raise RuntimeError(f"previous A0 source is incomplete: {case.source}")
        if not h192_screen_active(gpu):
            raise RuntimeError(f"192 A0 screen exited without a complete source: {case.source}")
        if not reported:
            emit("WAIT_A0_SOURCE", case=case.key, source=str(case.source))
            reported = True
        time.sleep(poll_seconds)
    source_ready(case, verify_weights=True)
    if reported:
        emit("A0_SOURCE_READY", case=case.key)


def gpu_busy(gpu: int) -> bool:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
         "--format=csv,noheader"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed on GPU {gpu}: {result.stderr.strip()}")
    return bool(re.search(r"(?m)^\s*\d+\s*$", result.stdout))


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(stream, mode)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def resource_slot(paths: tuple[Path, ...], poll_seconds: float, kind: str):
    """Hold one host-local resource slot until the child process exits."""
    import fcntl
    reported = False
    while True:
        for number, path in enumerate(paths):
            with path.open("a+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    if reported:
                        emit("SLOT_ACQUIRED", kind=kind, slot=number)
                    yield number
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                return
        if not reported:
            emit("WAIT_SLOT", kind=kind)
            reported = True
        time.sleep(poll_seconds)


@contextlib.contextmanager
def ram_slot(poll_seconds: float):
    # A slot limits admitted processes; the MemAvailable check below provides
    # the actual host-memory guard when some cases use much more RAM than others.
    with resource_slot(RAM_SLOT_PATHS, poll_seconds, "ram") as number:
        yield number


@contextlib.contextmanager
def heavy_slot(poll_seconds: float):
    with resource_slot(HEAVY_SLOT_PATHS, poll_seconds, "heavy") as number:
        yield number


def run_case(case: Case, gpu: int, output_root: Path, runner: Path,
             python: Path, poll_seconds: float, max_attempts: int) -> bool:
    if not source_ready(case):
        raise RuntimeError(f"source was ready at launch but is now incomplete: {case.source}")
    checkpoint = case.source / "token" / "best_val.pt"
    a0_sha = sha256(checkpoint)
    output = output_root / "cases" / case.key
    log = output_root / "queue_logs" / f"{case.slug}.log"
    if complete_output(output, case, a0_sha):
        emit("SKIP_VERIFIED_COMPLETE", case=case.key, output=str(output))
        return True
    if (output / "downstream_complete").exists():
        raise RuntimeError(f"invalid completion marker; preserving output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(python), str(runner), "--case-key", case.key,
               "--a0-run-dir", str(case.source), "--output-dir", str(output),
               "--gpu", str(gpu), "--selection-split", "test", "--resume"]
    if case.dataset in HEAVY_DATASETS:
        command.extend(["--extract-batch-size", "1"])
    gpu_lock = Path(f"/tmp/fpem_periodic_gpu_{gpu}.lock")
    heavy = case.dataset in HEAVY_DATASETS
    minimum_gib = HEAVY_MIN_AVAILABLE_GIB if heavy else LIGHT_MIN_AVAILABLE_GIB
    for attempt in range(1, max_attempts + 1):
        with file_lock(gpu_lock):
            reported_wait = False
            while True:
                # Heavy admission comes first. A heavy case waiting for a slot
                # must not occupy one of the RAM slots needed by a light case.
                with (heavy_slot(poll_seconds) if heavy
                      else contextlib.nullcontext()):
                    with ram_slot(poll_seconds):
                        with file_lock(ADMISSION_LOCK):
                            free_gib = available_gib()
                            busy = gpu_busy(gpu)
                            ready = not busy and free_gib >= minimum_gib
                            if ready:
                                emit("START", case=case.key, gpu=gpu,
                                     attempt=attempt, mem_available_gib=round(free_gib, 1),
                                     heavy=heavy)
                                handle = log.open("a", encoding="utf-8")
                                try:
                                    handle.write(f"\nSTART attempt={attempt} case={case.key} "
                                                 f"source={case.source} a0_sha256={a0_sha}\n")
                                    handle.flush()
                                    process = subprocess.Popen(
                                        command, stdout=handle, stderr=subprocess.STDOUT)
                                except Exception:
                                    handle.close()
                                    raise
                                # Keep new launches staggered until the
                                # preceding process has allocated memory.
                                try:
                                    code = process.wait(timeout=START_SPACING_SECONDS)
                                except subprocess.TimeoutExpired:
                                    code = None
                            elif not reported_wait:
                                emit("WAIT_RESOURCES", case=case.key, gpu=gpu,
                                     mem_available_gib=round(free_gib, 1),
                                     minimum_gib=minimum_gib, gpu_busy=busy)
                                reported_wait = True
                        if ready:
                            try:
                                if code is None:
                                    code = process.wait()
                                handle.write(f"EXIT attempt={attempt} code={code}\n")
                            finally:
                                handle.close()
                            break
                time.sleep(poll_seconds)
        if code == 0 and complete_output(output, case, a0_sha):
            emit("COMPLETE", case=case.key, gpu=gpu, attempt=attempt)
            return True
        emit("ATTEMPT_FAILED", case=case.key, gpu=gpu, attempt=attempt,
             exit_code=code, log=str(log))
        if attempt < max_attempts:
            time.sleep(max(60, poll_seconds) * attempt)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(H192_DATASET_GPU), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--repo-root", type=Path,
                        default=Path("/data/OuXiaoyu/Time-Series-Library-FPEM"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--audit-old-sources", action="store_true")
    cli = parser.parse_args()
    if cli.poll_seconds <= 0 or cli.max_attempts <= 0:
        parser.error("poll-seconds and max-attempts must be positive")
    repo = cli.repo_root.resolve()
    output_root = cli.output_root.resolve()
    runner = cli.runner or repo / "tools" / "run_fpem_token_eiil3_full.py"
    plan = make_plan(repo, cli.host)
    if cli.print_plan:
        for gpu, cases in plan.items():
            for case in cases:
                emit("PLAN", gpu=gpu, case=case.key, source=str(case.source))
        return 0
    if cli.audit_old_sources:
        previous = [case for cases in plan.values() for case in cases
                    if case.horizon in OLD_HORIZONS]
        for case in previous:
            if not source_ready(case):
                raise RuntimeError(f"incomplete previous A0 source: {case.source}")
        emit("OLD_A0_AUDIT_OK", host=cli.host, count=len(previous))
        return 0
    if not runner.is_file():
        raise FileNotFoundError(f"downstream runner is missing: {runner}")
    output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(Path(f"/tmp/fpem_periodic_downstream_{cli.host}_g{cli.gpu}.queue.lock"),
                   blocking=False):
        wait_for_gpu_a0(cli.gpu, cli.poll_seconds)
        failed = []
        for case in plan[cli.gpu]:
            try:
                wait_for_source(case, cli.gpu, cli.poll_seconds)
                if not run_case(case, cli.gpu, output_root, runner,
                                cli.python, cli.poll_seconds, cli.max_attempts):
                    failed.append(case.key)
            except Exception as exc:
                failed.append(case.key)
                emit("CASE_ERROR", case=case.key, error=repr(exc))
        emit("QUEUE_END", host=cli.host, gpu=cli.gpu,
             total=len(plan[cli.gpu]), failed=failed)
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
