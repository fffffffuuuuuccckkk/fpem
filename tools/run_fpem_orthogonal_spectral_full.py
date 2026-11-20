"""Restartable O3/O4 full matrix on one host; TEST-guided LR search.

Each run still selects its epoch on VAL, as the model trainer does.  Rank and
learning rates are then selected on TEST at the user's explicit request, so
these TEST results are exploratory/development results, not an unbiased final
evaluation.  No A0 checkpoint is trained or modified here.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/fpem_orthogonal_spectral_full_testlr_20261020"
OLD = ROOT / "results/fpem_orthogonal_spectral_decomposition"
FRACTIONS = (0.0625, 0.125, 0.25)
HEAD_LRS = (2e-5, 5e-5, 1e-4, 2e-4)
BASIS_LRS = (5e-5, 1e-4, 2e-4)
BASE_HEAD_LR = 5e-5
BASE_BASIS_LR = 1e-4
LOCK = threading.Lock()


def tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def trial_path(case: str, method: str, fraction: float,
               head_lr: float, basis_lr: float) -> Path:
    return OUTPUT / case / method / (
        f"r{tag(fraction)}_h{tag(head_lr)}_b{tag(basis_lr)}")


def load_result(path: Path):
    if (path / "run_complete").is_file() and (path / "result.json").is_file():
        return json.loads((path / "result.json").read_text(encoding="utf-8")), path
    pointer = path / "reused_from.json"
    if pointer.is_file():
        source = Path(json.loads(pointer.read_text(encoding="utf-8"))["source"])
        if (source / "run_complete").is_file() and (source / "result.json").is_file():
            return json.loads((source / "result.json").read_text(encoding="utf-8")), source
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["pred_len"] = int(row["pred_len"])
        row["gpu"] = int(row["gpu"])
        row["case"] = f'{row["backbone"]}_{row["dataset"]}_{row["pred_len"]}'
        config = ROOT / row["base_run_config"]
        if not config.is_file() and row["case"] == "itransformer_ExchangeRate_24":
            template = ROOT / (
                "results/fpem_no_future_anchor_itransformer_search_20261012/"
                "itransformer/ExchangeRate/pred_48/A0/run_config.json")
            saved = json.loads(template.read_text(encoding="utf-8"))
            saved["pred_len"] = 24
            saved["reference_checkpoint"] = (
                "results/foil_shifts_a0_protocol_20261011/"
                "itransformer/ExchangeRate/pred_24/shared_reference.pt")
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        if not config.is_file():
            raise FileNotFoundError(config)
        saved = json.loads(config.read_text(encoding="utf-8"))
        if (saved["backbone"].lower(), saved["dataset_name"], int(saved["pred_len"])) != (
                row["backbone"], row["dataset"], row["pred_len"]):
            raise ValueError(f'base config mismatch: {row["case"]}: {config}')
        # A few newer S4 run_configs name a decomposition unknown to the older
        # secondary-server model. This trainer only uses the A0 encoder/head;
        # normalize the otherwise unused constructor mode without changing A0.
        if saved.get("decomposition_type") not in (
                "projection", "signed_gate", "complementary_gate"):
            saved["decomposition_type"] = "complementary_gate"
            config = OUTPUT / "configs" / f'{row["case"]}_a0_compatible.json'
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
            row["base_run_config"] = str(config.relative_to(ROOT))
        reference = Path(saved["reference_checkpoint"])
        if not reference.is_absolute():
            reference = ROOT / reference
        if not reference.is_file():
            raise FileNotFoundError(reference)
        data_root = Path(saved["root_path"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        if not (data_root / saved["data_path"]).is_file():
            raise FileNotFoundError(data_root / saved["data_path"])
        row["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    if len({row["case"] for row in rows}) != len(rows):
        raise ValueError("duplicate manifest case")
    return rows


def one(row: dict, method: str, fraction: float,
        head_lr: float, basis_lr: float) -> tuple[dict, Path]:
    case = row["case"]
    path = trial_path(case, method, fraction, head_lr, basis_lr)
    old_path = OLD / case / method / path.name
    existing = load_result(path)
    if existing is None and (old := load_result(old_path)) is not None:
        if old[0]["reference_sha256"] == row["reference_sha256"]:
            path.mkdir(parents=True, exist_ok=True)
            (path / "reused_from.json").write_text(
                json.dumps({"source": str(old_path)}) + "\n", encoding="utf-8")
            existing = old
    if existing is not None:
        if existing[0]["reference_sha256"] != row["reference_sha256"]:
            raise ValueError(f"A0 reference mismatch: {path}")
        print(f"SKIP/REUSE {case} {method} r={fraction} h={head_lr} b={basis_lr}", flush=True)
        return existing
    path.mkdir(parents=True, exist_ok=True)
    spectrum = OUTPUT / case / f"train_spectrum_r{tag(fraction)}.pt"
    cmd = [sys.executable, "tools/run_fpem_orthogonal_spectral.py",
           "--base_run_config", row["base_run_config"], "--output", str(path),
           "--method", method, "--variation_fraction", str(fraction),
           "--lr_head", str(head_lr), "--lr_basis", str(basis_lr),
           "--lr_backbone", "2e-5", "--seed", "2021", "--epochs", "10",
           "--gpu", "0", "--spectrum_cache", str(spectrum),
           "--allow_legacy_reference_signature"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(row["gpu"])
    for attempt in range(1, 4):
        print(f"START {case} {method} r={fraction} h={head_lr} b={basis_lr} "
              f"gpu={row['gpu']} attempt={attempt}", flush=True)
        with (path / "run.log").open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND " + " ".join(cmd) + "\n")
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        result = load_result(path)
        if code == 0 and result is not None:
            if result[0]["reference_sha256"] != row["reference_sha256"]:
                raise ValueError(f"A0 reference changed during training: {path}")
            print(f"DONE {case} {method} r={fraction} h={head_lr} b={basis_lr}", flush=True)
            return result
        print(f"RETRY {case} {method} exit={code}; wait 60s", flush=True)
        time.sleep(60)
    (path / "failed_after_retries").touch()
    raise RuntimeError(f"failed three times: {path}")


def test_mse(trial: tuple[dict, Path]) -> float:
    return float(trial[0]["test"]["mse"])


def run_case(row: dict) -> None:
    case = row["case"]
    if (OUTPUT / case / "case_complete").is_file():
        print(f"CASE SKIP {case}", flush=True)
        return
    selected = []
    for method in ("O3", "O4"):
        trials = [
            (one(row, method, fraction, BASE_HEAD_LR, BASE_BASIS_LR),
             fraction, BASE_HEAD_LR, BASE_BASIS_LR)
            for fraction in FRACTIONS
        ]
        best = min(trials, key=lambda trial: test_mse(trial[0]))
        rank = best[1]
        for head_lr in HEAD_LRS:
            if head_lr == BASE_HEAD_LR:
                continue
            trials.append((one(row, method, rank, head_lr, BASE_BASIS_LR),
                           rank, head_lr, BASE_BASIS_LR))
        best = min(trials, key=lambda trial: test_mse(trial[0]))
        if method == "O4":
            rank, head_lr = best[1], best[2]
            for basis_lr in BASIS_LRS:
                if basis_lr == BASE_BASIS_LR:
                    continue
                trials.append((one(row, method, rank, head_lr, basis_lr),
                               rank, head_lr, basis_lr))
            best = min(trials, key=lambda trial: test_mse(trial[0]))
        selected.append({
            "case": case, "method": method, "rank_fraction": best[1],
            "lr_head": best[2], "lr_basis": best[3],
            "reference_sha256": row["reference_sha256"],
            "val_mse": best[0][0]["val"]["mse"],
            "val_mae": best[0][0]["val"]["mae"],
            "test_mse": best[0][0]["test"]["mse"],
            "test_mae": best[0][0]["test"]["mae"],
            "best_epoch_selected_by_val": best[0][0]["best_epoch"],
            "source": str(best[0][1]),
        })
    write_csv(OUTPUT / case / "selected_by_test.csv", selected)
    (OUTPUT / case / "case_complete").touch()
    print(f"CASE COMPLETE {case}", flush=True)


def summarize(rows: list[dict]) -> None:
    with LOCK:
        selected = []
        status = []
        for row in rows:
            case = row["case"]
            path = OUTPUT / case / "selected_by_test.csv"
            complete = (OUTPUT / case / "case_complete").is_file()
            status.append({"case": case, "gpu": row["gpu"], "complete": int(complete),
                           "reference_sha256": row["reference_sha256"]})
            if complete and path.is_file():
                with path.open(newline="", encoding="utf-8") as stream:
                    selected.extend(csv.DictReader(stream))
        write_csv(OUTPUT / "status.csv", status)
        write_csv(OUTPUT / "selected_by_test.csv", selected)
        lines = ["# O3/O4 full matrix — TEST-guided learning-rate search", "",
                 "Checkpoint epoch is selected using VAL MSE. Rank and learning rates",
                 "are selected using TEST MSE at the user's request. Thus TEST is a",
                 "development set here; these numbers are not unbiased final estimates.", "",
                 f"Completed cases: {sum(int(r['complete']) for r in status)}/{len(rows)}",
                 f"Completed selected method rows: {len(selected)}/{2*len(rows)}", "",
                 "A0 checkpoints are reused, never retrained. O3/O4 use TRAIN-only",
                 "predictive spectrum and the prior frozen-backbone 10-epoch protocol."]
        (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--validate-only"):
        raise SystemExit("usage: run_fpem_orthogonal_spectral_full.py MANIFEST.csv [--validate-only]")
    rows = load_manifest(ROOT / sys.argv[1])
    if len(sys.argv) == 3:
        import torch
        identity_fields = ("backbone", "dataset_name", "data_path", "seq_len",
                           "pred_len", "d_model", "d_ff", "n_heads", "e_layers",
                           "dropout", "seed", "warmup_epochs", "enc_in")
        for row in rows:
            saved = json.loads((ROOT / row["base_run_config"]).read_text(encoding="utf-8"))
            reference = Path(saved["reference_checkpoint"])
            if not reference.is_absolute():
                reference = ROOT / reference
            payload = torch.load(reference, map_location="cpu", weights_only=False)
            signature = payload.get("signature")
            if (not isinstance(signature, dict)
                    or any(signature.get(field) != saved.get(field)
                           for field in identity_fields)):
                raise ValueError(f'preflight A0 identity mismatch: {row["case"]}')
            del payload
        print(f"VALIDATED {len(rows)} cases; GPUs "
              + ", ".join(f"{gpu}:{sum(row['gpu']==gpu for row in rows)}" for gpu in range(4)))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "manifest.csv", rows)
    (OUTPUT / "search_protocol.json").write_text(json.dumps({
        "methods": ["O3", "O4"], "rank_fractions": FRACTIONS,
        "head_learning_rates": HEAD_LRS, "basis_learning_rates_O4": BASIS_LRS,
        "rank_and_lr_selection": "TEST MSE (exploratory)",
        "checkpoint_epoch_selection": "VAL MSE", "epochs": 10,
        "train_backbone": False, "a0_policy": "reuse existing checkpoint only",
        "manifest": sys.argv[1], "cases": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    summarize(rows)
    lanes = {gpu: [] for gpu in range(4)}
    for row in rows:
        lanes[row["gpu"]].append(row)
    def lane(gpu: int) -> None:
        for row in lanes[gpu]:
            try:
                run_case(row)
            except Exception as error:
                print(f"CASE FAILED {row['case']}: {error!r}", flush=True)
                (OUTPUT / row["case"] / "case_failed").touch()
            finally:
                summarize(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lane, gpu) for gpu in range(4)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summarize(rows)


if __name__ == "__main__":
    main()
