#!/usr/bin/env python3
"""Post-hoc semantic analysis of TRAIN-only predictive environments.

This tool never changes an environment assignment and never reads validation/test
targets.  It joins the exact final-stage TRAIN assignment saved by an experiment
to calendar and input-window-only physical summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate")
HORIZONS = (96, 336, 720)
ENV_NAMES = ("Env-1", "Env-2", "Env-3")
ORDER = {
    "season": ["Winter", "Transition", "Summer"],
    "day_night": ["Daytime", "Nighttime"],
    "weekday_weekend": ["Weekday", "Weekend"],
    "day_block": ["00-06", "06-12", "12-18", "18-24"],
    "temperature_regime": ["Low", "Medium", "High"],
    "humidity_regime": ["Low", "Medium", "High"],
    "wind_regime": ["Low", "Medium", "High"],
    "pressure_regime": ["Low", "Medium", "High"],
    "load_regime": ["Low", "Medium", "High"],
    "ot_regime": ["Low", "Medium", "High"],
    "volatility_regime": ["Low", "Medium", "High"],
}


def jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_best_configs(results_root: Path):
    records = []
    for path in results_root.glob("**/fpem_no_future_anchor_patchtst_search/patchtst/*/pred_*/best_config.json"):
        cfg = json.loads(path.read_text())
        dataset = cfg.get("dataset")
        pred_len = int(cfg.get("pred_len", 0))
        if dataset not in DATASETS or pred_len not in HORIZONS:
            continue
        run_name = Path(cfg["metrics_path"]).parts[-3]
        run_dir = path.parent / run_name / "A2"
        stages = list(run_dir.glob("environment_gradient_stage_*.npz"))
        if not stages:
            continue
        final_stage = max(stages, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
        records.append({"dataset": dataset, "pred_len": pred_len, "best_config": path,
                        "config": cfg, "run_dir": run_dir, "environment_file": final_stage})
    # If synchronized mirrors duplicate a case, prefer primary, then newest file.
    chosen = {}
    for record in records:
        key = (record["dataset"], record["pred_len"])
        score = ("server2_imports" not in str(record["best_config"]), record["environment_file"].stat().st_mtime)
        if key not in chosen or score > chosen[key][0]:
            chosen[key] = (score, record)
    return [chosen[key][1] for key in sorted(chosen)]


def dataset_csv(repo: Path, dataset: str) -> Path:
    if dataset == "Weather":
        return repo / "dataset/all_datasets/weather/weather.csv"
    if dataset == "ExchangeRate":
        return repo / "dataset/all_datasets/exchange_rate/exchange_rate.csv"
    return repo / "dataset/all_datasets/ETT-small" / f"{dataset}.csv"


def season(month):
    return np.select([np.isin(month, [12, 1, 2]), np.isin(month, [6, 7, 8])],
                     ["Winter", "Summer"], default="Transition")


def rolling_mean(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    cumulative = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
    counts = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    total = cumulative[starts + length] - cumulative[starts]
    count = counts[starts + length] - counts[starts]
    return total / np.maximum(count, 1)


def rolling_volatility(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    # Aggregate standard deviation of input-window first differences across all currencies.
    diff = np.diff(np.asarray(values, dtype=np.float64), axis=0)
    sq = diff * diff
    c1 = np.vstack([np.zeros((1, diff.shape[1])), np.cumsum(diff, axis=0)])
    c2 = np.vstack([np.zeros((1, diff.shape[1])), np.cumsum(sq, axis=0)])
    n = max(length - 1, 1)
    total = c1[starts + n] - c1[starts]
    total2 = c2[starts + n] - c2[starts]
    variance = np.maximum(total2 / n - (total / n) ** 2, 0)
    return np.sqrt(variance).mean(1)


def regimes(values):
    q33, q67 = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.select([values <= q33, values <= q67], ["Low", "Medium"], default="High")
    return labels, [float(q33), float(q67)]


def match_column(columns, exact=(), contains=()):
    lowered = {str(c).lower(): c for c in columns}
    for name in exact:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for token in contains:
        matches = [c for c in columns if token.lower() in str(c).lower()]
        if matches:
            return matches[0]
    raise KeyError(f"cannot map column from exact={exact}, contains={contains}; columns={list(columns)}")


def build_semantics(repo: Path, dataset: str, pred_len: int, sample_ids: np.ndarray, seq_len: int):
    csv_path = dataset_csv(repo, dataset)
    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw["date"])
    if dataset.startswith("ETTh"):
        train_rows = 12 * 30 * 24
    elif dataset.startswith("ETTm"):
        train_rows = 12 * 30 * 24 * 4
    else:
        train_rows = int(len(raw) * 0.7)
    expected = train_rows - seq_len - pred_len + 1
    if len(sample_ids) != expected or not np.array_equal(sample_ids, np.arange(expected)):
        raise RuntimeError(
            f"sample alignment failed for {dataset}-{pred_len}: IDs={len(sample_ids)}, expected={expected}, "
            f"chronological={np.array_equal(sample_ids, np.arange(len(sample_ids)))}"
        )
    end_rows = sample_ids + seq_len - 1
    timestamps = raw.loc[end_rows, "date"].reset_index(drop=True)
    semantic = pd.DataFrame({
        "sample_id": sample_ids,
        "window_start_timestamp": raw.loc[sample_ids, "date"].to_numpy(),
        "window_end_timestamp": timestamps.to_numpy(),
        "month": timestamps.dt.month.astype(str).str.zfill(2),
        "season": season(timestamps.dt.month.to_numpy()),
        "hour": timestamps.dt.hour.astype(str).str.zfill(2),
        "day_night": np.where(timestamps.dt.hour.between(6, 17), "Daytime", "Nighttime"),
        "weekday_weekend": np.where(timestamps.dt.weekday < 5, "Weekday", "Weekend"),
    })
    metadata = {"csv": str(csv_path), "train_rows": train_rows, "seq_len": seq_len,
                "pred_len": pred_len, "physical_columns": {}, "quantiles": {},
                "alignment": "sample i uses raw TRAIN rows [i, i+seq_len); label timestamp is i+seq_len-1"}
    continuous = {}
    if dataset == "Weather":
        mapping = {
            "temperature": match_column(raw.columns, exact=("T (degC)",), contains=("temperature", "temp")),
            "humidity": match_column(raw.columns, exact=("rh (%)",), contains=("humidity", "rh (")),
            "wind": match_column(raw.columns, exact=("wv (m/s)",), contains=("wind", "wv (")),
            "pressure": match_column(raw.columns, exact=("p (mbar)",), contains=("pressure", "p (")),
        }
        metadata["physical_columns"] = mapping
        for key, column in mapping.items():
            values = raw[column].to_numpy(float)
            # Jena Weather uses -999/-9999 sentinels for missing sensor values.
            invalid = (~np.isfinite(values)) | (values <= -900)
            metadata.setdefault("invalid_sensor_values", {})[key] = int(invalid[:train_rows].sum())
            values = values.copy(); values[invalid] = np.nan
            mean = rolling_mean(values, sample_ids, seq_len)
            tail = values[end_rows]
            semantic[f"{key}_window_mean"] = mean
            semantic[f"{key}_window_tail"] = tail
            semantic[f"{key}_regime"], metadata["quantiles"][key] = regimes(mean)
            continuous[key] = mean
    elif dataset.startswith("ETT"):
        load_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        missing = [c for c in load_columns + ["OT"] if c not in raw.columns]
        if missing:
            raise KeyError(f"missing actual ETT variables: {missing}")
        train_values = raw.loc[: train_rows - 1, load_columns].to_numpy(float)
        mu, sigma = train_values.mean(0), train_values.std(0) + 1e-8
        z = np.abs((raw[load_columns].to_numpy(float) - mu) / sigma).mean(1)
        aggregate = rolling_mean(z, sample_ids, seq_len)
        ot = rolling_mean(raw["OT"].to_numpy(float), sample_ids, seq_len)
        semantic["aggregate_load_window_mean"] = aggregate
        semantic["ot_window_mean"] = ot
        semantic["load_regime"], metadata["quantiles"]["aggregate_load"] = regimes(aggregate)
        semantic["ot_regime"], metadata["quantiles"]["OT"] = regimes(ot)
        metadata["physical_columns"] = {"aggregate_load": load_columns, "OT": "OT"}
        continuous.update({"aggregate_load": aggregate, "OT": ot})
        if dataset.startswith("ETTm"):
            hour = timestamps.dt.hour.to_numpy()
            semantic["day_block"] = np.select([hour < 6, hour < 12, hour < 18],
                                                ["00-06", "06-12", "12-18"], default="18-24")
    else:
        value_columns = [c for c in raw.columns if c != "date"]
        volatility = rolling_volatility(raw[value_columns].to_numpy(float), sample_ids, seq_len)
        semantic["year"] = timestamps.dt.year.astype(str)
        semantic["quarter"] = "Q" + timestamps.dt.quarter.astype(str)
        semantic["month_of_year"] = timestamps.dt.month.astype(str).str.zfill(2)
        semantic["volatility_window"] = volatility
        semantic["volatility_regime"], metadata["quantiles"]["volatility"] = regimes(volatility)
        metadata["physical_columns"] = {"volatility": value_columns}
        continuous["volatility"] = volatility
    return semantic, metadata, continuous


def categorical_columns(dataset, semantic):
    if dataset == "Weather":
        wanted = ["season", "month", "hour", "day_night", "temperature_regime",
                  "humidity_regime", "wind_regime", "pressure_regime"]
    elif dataset.startswith("ETT"):
        wanted = ["season", "month", "hour", "day_night", "weekday_weekend",
                  "load_regime", "ot_regime"]
        if dataset.startswith("ETTm"):
            wanted.append("day_block")
    else:
        wanted = ["year", "quarter", "month_of_year", "volatility_regime"]
    return [name for name in wanted if name in semantic]


def ordered_groups(name, values):
    observed = list(pd.unique(values.astype(str)))
    if name in ORDER:
        return [group for group in ORDER[name] if group in observed]
    def key(value):
        match = re.search(r"\d+", value)
        return (int(match.group()) if match else 10**9, value)
    return sorted(observed, key=key)


def contingency(labels, hard, groups, env_num=3):
    index = {name: i for i, name in enumerate(groups)}
    codes = np.array([index[str(v)] for v in labels], dtype=np.int64)
    table = np.bincount(codes * env_num + hard, minlength=len(groups) * env_num)
    return table.reshape(len(groups), env_num), codes


def nmi_from_tables(tables):
    tables = np.asarray(tables, dtype=np.float64)
    if tables.ndim == 2:
        tables = tables[None]
    total = tables.sum((1, 2), keepdims=True)
    p = tables / np.maximum(total, 1)
    pr = p.sum(2, keepdims=True)
    pc = p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log(p / np.maximum(pr * pc, 1e-300)), 0)
        hr = -np.where(pr > 0, pr * np.log(pr), 0).sum((1, 2))
        hc = -np.where(pc > 0, pc * np.log(pc), 0).sum((1, 2))
    return 2 * term.sum((1, 2)) / np.maximum(hr + hc, 1e-12)


def permutation_nmi(table, repeats, seed):
    rng = np.random.default_rng(seed)
    try:
        from scipy.stats import random_table
        sampled = random_table.rvs(table.sum(1), table.sum(0), size=repeats, random_state=rng)
        null = nmi_from_tables(sampled)
        method = "exact permutation-null contingency sampling with fixed margins"
    except Exception:
        # Exact label permutations fallback.
        labels = np.concatenate([
            np.repeat(row, int(table[row, col]))
            for row in range(table.shape[0]) for col in range(table.shape[1])
        ])
        env = np.concatenate([
            np.repeat(col, int(table[row, col]))
            for row in range(table.shape[0]) for col in range(table.shape[1])
        ])
        null = []
        for _ in range(repeats):
            perm = rng.permutation(env)
            sampled = np.bincount(labels * table.shape[1] + perm,
                                  minlength=table.size).reshape(table.shape)
            null.append(nmi_from_tables(sampled)[0])
        null = np.asarray(null)
        method = "explicit environment-label permutation"
    return null, method


def association_metrics(labels, hard, groups, repeats, seed):
    from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
    from scipy.stats import chi2_contingency
    table, codes = contingency(labels, hard, groups)
    observed = normalized_mutual_info_score(codes, hard)
    null, method = permutation_nmi(table, repeats, seed)
    null_mean, null_std = float(null.mean()), float(null.std())
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    cramer = math.sqrt(chi2 / max(n * min(table.shape[0] - 1, table.shape[1] - 1), 1))
    return table, codes, {
        "nmi": float(observed), "ari": float(adjusted_rand_score(codes, hard)),
        "mutual_information": float(mutual_info_score(codes, hard)), "cramers_v": float(cramer),
        "permutation_repeats": int(repeats), "null_mean_nmi": null_mean,
        "null_std_nmi": null_std, "empirical_p_value": float((1 + (null >= observed).sum()) / (repeats + 1)),
        "z_score": float((observed - null_mean) / max(null_std, 1e-12)),
        "percentile": float(100 * (null < observed).mean()), "permutation_method": method,
    }


def extract_representation_metrics(path: Path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    flat = {}
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    walk(value)
                elif isinstance(value, (int, float)):
                    flat.setdefault(key, value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    walk(payload)
    aliases = {
        "var_env_accuracy": ("var_env_accuracy_argmax", "var_acc", "var_env_acc"),
        "inv_env_accuracy": ("inv_env_accuracy_argmax", "inv_acc", "inv_env_acc"),
        "var_env_soft_ce": ("var_env_soft_ce", "var_soft_ce"),
        "inv_env_soft_ce": ("inv_env_soft_ce", "inv_soft_ce"),
    }
    result = {}
    for target, names in aliases.items():
        for name in names:
            if name in flat:
                result[target] = float(flat[name]); break
    if "var_env_accuracy" in result and "inv_env_accuracy" in result:
        result["env_accuracy_gap_var_minus_inv"] = result["var_env_accuracy"] - result["inv_env_accuracy"]
    return result


def save_figure(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def heatmap(matrix, rows, title, base, center=None, cmap="viridis", fmt=".2f"):
    import matplotlib.pyplot as plt
    width = max(5.2, 0.55 * matrix.shape[1] + 2.2)
    height = max(3.4, 0.38 * matrix.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    values = np.asarray(matrix, dtype=float)
    if center is None:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    else:
        bound = max(abs(np.nanmin(values) - center), abs(np.nanmax(values) - center), 1e-12)
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=center - bound, vmax=center + bound)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.04)
    ax.set_xticks(range(values.shape[1]), ENV_NAMES[:values.shape[1]])
    ax.set_yticks(range(values.shape[0]), rows)
    threshold = (np.nanmin(values) + np.nanmax(values)) / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, format(values[row, col], fmt), ha="center", va="center",
                    fontsize=8, color="white" if values[row, col] < threshold else "black")
    ax.set_title(title); ax.set_xlabel("Predictive environment"); ax.set_ylabel("Semantic regime")
    save_figure(fig, base); plt.close(fig)


def plot_timeline(assignments, semantic, dataset, base):
    import matplotlib.pyplot as plt
    timestamp = pd.to_datetime(semantic["window_end_timestamp"])
    q = assignments[["q_env1", "q_env2", "q_env3"]].to_numpy()
    take = np.unique(np.linspace(0, len(q) - 1, min(len(q), 6000)).astype(int))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for k in range(3):
        ax.plot(timestamp.iloc[take], q[take, k], lw=0.75, alpha=0.8, label=ENV_NAMES[k])
    if dataset == "Weather":
        summer = semantic["season"].to_numpy() == "Summer"
        winter = semantic["season"].to_numpy() == "Winter"
        ax.fill_between(timestamp.iloc[take], 0, 1, where=summer[take], color="#f4a261", alpha=.07,
                        transform=ax.get_xaxis_transform(), label="Summer")
        ax.fill_between(timestamp.iloc[take], 0, 1, where=winter[take], color="#457b9d", alpha=.07,
                        transform=ax.get_xaxis_transform(), label="Winter")
    ax.set_ylim(0, 1); ax.set_ylabel("Soft assignment probability"); ax.set_xlabel("TRAIN time")
    ax.set_title(f"{dataset}: predictive environment timeline"); ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=.15); save_figure(fig, base); plt.close(fig)


def plot_hour_profile(assignments, semantic, dataset, base):
    import matplotlib.pyplot as plt
    frame = assignments[["q_env1", "q_env2", "q_env3"]].copy()
    frame["hour"] = semantic["hour"].astype(int)
    profile = frame.groupby("hour").mean().reindex(range(24))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k in range(3):
        ax.plot(profile.index, profile.iloc[:, k], marker="o", ms=3, label=ENV_NAMES[k])
    ax.set_xticks(range(0, 24, 2)); ax.set_ylim(0, 1); ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean soft assignment"); ax.set_title(f"{dataset}: 24-hour environment profile")
    ax.legend(); ax.grid(alpha=.2); save_figure(fig, base); plt.close(fig)
    profile.to_csv(base.with_suffix(".csv"), index_label="hour")


def plot_physical(assignments, continuous, dataset, base):
    import matplotlib.pyplot as plt
    hard = assignments["hard_env_raw"].to_numpy()
    frames = []
    for name, values in continuous.items():
        frame = pd.DataFrame({"value": values, "variable": name,
                              "environment": [ENV_NAMES[i] for i in hard]})
        if len(frame) > 15000:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, 15000).astype(int)]
        frames.append(frame)
    if not frames:
        return
    long = pd.concat(frames, ignore_index=True)
    variables = list(continuous)
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 4.2), squeeze=False)
    for ax, variable in zip(axes[0], variables):
        subset = long[long.variable == variable]
        arrays = [subset.loc[subset.environment == name, "value"].to_numpy() for name in ENV_NAMES]
        violin = ax.violinplot(arrays, positions=np.arange(3), showmeans=False,
                               showmedians=True, showextrema=False)
        for index, body in enumerate(violin["bodies"]):
            body.set_facecolor(plt.get_cmap("tab10")(index)); body.set_alpha(.65)
        ax.set_xticks(range(3), ENV_NAMES)
        ax.set_title(variable); ax.set_xlabel(""); ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"{dataset}: input-window physical state by predictive environment", y=1.02)
    save_figure(fig, base); plt.close(fig)


def analyze_case(repo, record, output_root, repeats, seed):
    dataset, pred_len = record["dataset"], record["pred_len"]
    case_dir = output_root / dataset / f"pred_{pred_len}"
    figures = case_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    env_file = record["environment_file"]
    env = np.load(env_file)
    sample_ids = env["sample_id"].astype(np.int64)
    q = env["q"].astype(np.float64)
    if q.shape[1] != 3 or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise RuntimeError(f"invalid K=3 q in {env_file}")
    hard = q.argmax(1)
    assignments = pd.DataFrame({"sample_id": sample_ids, "hard_env_raw": hard,
                                "q_env1": q[:, 0], "q_env2": q[:, 1], "q_env3": q[:, 2],
                                "g_scale": env["g_scale"], "g_bias": env["g_bias"],
                                "train_sample_mse": env["sample_mse"], "train_sample_mae": env["sample_mae"]})
    run_cfg_path = record["run_dir"].parent / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}
    seq_len = int(run_cfg.get("seq_len", 96))
    semantic, metadata, continuous = build_semantics(repo, dataset, pred_len, sample_ids, seq_len)
    if not np.array_equal(assignments.sample_id, semantic.sample_id):
        raise RuntimeError("semantic/environment sample IDs diverged")
    assignments.insert(1, "window_end_timestamp", semantic["window_end_timestamp"])
    assignments.to_csv(case_dir / "sample_environment_assignments.csv", index=False)
    semantic.to_csv(case_dir / "semantic_labels.csv", index=False)

    metrics, soft_rows, composition_rows, enrichment_rows = [], [], [], []
    for position, variable in enumerate(categorical_columns(dataset, semantic)):
        values = semantic[variable].astype(str).to_numpy()
        groups = ordered_groups(variable, semantic[variable])
        table, codes, metric = association_metrics(values, hard, groups, repeats, seed + position * 1009)
        metric.update({"dataset": dataset, "pred_len": pred_len, "semantic_variable": variable})
        metrics.append(metric)
        soft = np.vstack([q[codes == i].mean(0) for i in range(len(groups))])
        p_env_given_group = table / np.maximum(table.sum(1, keepdims=True), 1)
        p_group_given_env = table / np.maximum(table.sum(0, keepdims=True), 1)
        p_group = table.sum(1) / table.sum()
        enrich = p_group_given_env / np.maximum(p_group[:, None], 1e-12)
        for i, group in enumerate(groups):
            for k in range(3):
                common = {"dataset": dataset, "pred_len": pred_len, "semantic_variable": variable,
                          "semantic_group": group, "environment": ENV_NAMES[k], "environment_raw_id": k}
                soft_rows.append({**common, "mean_soft_assignment": soft[i, k], "count": int(table[i].sum())})
                composition_rows.extend([
                    {**common, "probability_type": "P(environment|semantic_group)", "probability": p_env_given_group[i, k]},
                    {**common, "probability_type": "P(semantic_group|environment)", "probability": p_group_given_env[i, k]},
                ])
                enrichment_rows.append({**common, "enrichment": enrich[i, k],
                                        "p_semantic_given_environment": p_group_given_env[i, k],
                                        "p_semantic_marginal": p_group[i]})
        safe = variable.replace("/", "_")
        heatmap(soft, groups, f"{dataset} pred={pred_len}: mean q by {variable}",
                figures / f"soft_assignment_{safe}")
        heatmap(enrich, groups, f"{dataset} pred={pred_len}: enrichment by {variable}",
                figures / f"enrichment_{safe}", center=1, cmap="RdBu_r")

    representation = extract_representation_metrics(record["run_dir"] / "metrics_and_diagnostics.json")
    for metric in metrics:
        metric.update(representation)
    pd.DataFrame(soft_rows).to_csv(case_dir / "semantic_soft_assignment.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(case_dir / "environment_composition.csv", index=False)
    pd.DataFrame(enrichment_rows).to_csv(case_dir / "environment_enrichment.csv", index=False)
    metrics_payload = {"provenance": {
        "best_config": str(record["best_config"]), "best_config_values": record["config"],
        "environment_source": str(env_file), "environment_source_sha256": sha256(env_file),
        "assignment_source": "exact saved final-stage TRAIN assignment; no retraining/re-inference",
        "semantic_source": "calendar and input-window-only TRAIN values; no validation/test/future target",
        **metadata}, "representation_metrics": representation, "semantic_metrics": metrics}
    (case_dir / "environment_semantic_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=jsonable) + "\n")
    lines = [f"{dataset} pred_len={pred_len}", f"Environment source: {env_file}",
             f"Samples: {len(sample_ids)}; exact alignment: PASS", f"Physical columns: {metadata['physical_columns']}", ""]
    for metric in sorted(metrics, key=lambda item: item["nmi"], reverse=True):
        lines.append(f"{metric['semantic_variable']}: NMI={metric['nmi']:.6f}, p={metric['empirical_p_value']:.6g}, "
                     f"z={metric['z_score']:.3f}, ARI={metric['ari']:.6f}, Cramer's V={metric['cramers_v']:.6f}")
    (case_dir / "environment_semantic_metrics.txt").write_text("\n".join(lines) + "\n")
    if dataset in ("Weather", "ETTm1"):
        plot_timeline(assignments, semantic, dataset, figures / "environment_timeline")
    if dataset.startswith("ETTm"):
        plot_hour_profile(assignments, semantic, dataset, figures / "environment_24hour_profile")
    plot_physical(assignments, continuous, dataset, figures / "physical_distribution_by_environment")
    return metrics, enrichment_rows, metrics_payload["provenance"]


def summarize(output_root: Path, all_metrics, all_enrichment, provenance):
    import matplotlib.pyplot as plt
    metrics = pd.DataFrame(all_metrics)
    enrichment = pd.DataFrame(all_enrichment)
    metrics.to_csv(output_root / "semantic_analysis_summary.csv", index=False)
    enrichment.to_csv(output_root / "semantic_enrichment_summary.csv", index=False)
    mapping = {"Weather": ["season", "day_night", "temperature_regime", "humidity_regime", "wind_regime", "pressure_regime"],
               "ETTh1": ["season", "day_night", "load_regime", "ot_regime"],
               "ETTh2": ["season", "day_night", "load_regime", "ot_regime"],
               "ETTm1": ["season", "day_night", "day_block", "load_regime", "ot_regime"],
               "ETTm2": ["season", "day_night", "day_block", "load_regime", "ot_regime"],
               "ExchangeRate": ["year", "quarter", "month_of_year", "volatility_regime"]}
    columns = sorted({v for values in mapping.values() for v in values})
    summary = pd.DataFrame(index=DATASETS, columns=columns, dtype=float)
    for dataset in DATASETS:
        for variable in mapping[dataset]:
            values = metrics[(metrics.dataset == dataset) & (metrics.semantic_variable == variable)].nmi
            if len(values): summary.loc[dataset, variable] = values.mean()
    summary.to_csv(output_root / "nmi_summary_matrix.csv")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    masked = np.ma.masked_invalid(summary.to_numpy(float))
    image = ax.imshow(masked, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=.03, pad=.03)
    ax.set_xticks(range(len(summary.columns)), summary.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(summary.index)), summary.index)
    for row in range(masked.shape[0]):
        for col in range(masked.shape[1]):
            if not masked.mask[row, col]:
                ax.text(col, row, f"{masked[row, col]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Predictive environment – semantic association (mean NMI across horizons)")
    ax.set_xlabel("Semantic variable"); ax.set_ylabel("Dataset")
    save_figure(fig, output_root / "nmi_summary"); plt.close(fig)
    strongest = metrics.sort_values("nmi", ascending=False).groupby("dataset", sort=False).head(1)
    top_enrich = enrichment.sort_values("enrichment", ascending=False).groupby(["dataset", "pred_len"], sort=False).head(3)
    lines = ["Predictive Environment Semantic Analysis", "",
             "All semantic labels are post-hoc. Environments are the exact saved final-stage TRAIN assignments.",
             "No validation/test/future target is used for semantic labels. NMI is association, not environment quality.", "",
             "Strongest semantic association per dataset:"]
    for _, row in strongest.iterrows():
        lines.append(f"- {row.dataset}: {row.semantic_variable}, mean/run NMI={row.nmi:.6f}, p={row.empirical_p_value:.6g}, z={row.z_score:.2f}")
    lines += ["", "Largest enrichments per case:"]
    for _, row in top_enrich.iterrows():
        lines.append(f"- {row.dataset} pred={int(row.pred_len)} {row.environment}: {row.semantic_variable}={row.semantic_group}, enrichment={row.enrichment:.3f}")
    (output_root / "semantic_analysis_summary.txt").write_text("\n".join(lines) + "\n")
    readme = ["FPEM Predictive Environment Semantic Analysis", "",
              "Protocol", "- Exact saved final-stage q from each current no-future/no-anchor best run.",
              "- TRAIN samples remain chronological and sample_id is asserted to equal 0..N-1.",
              "- Calendar labels use the input-window end timestamp.",
              "- Physical regimes use input-window means and 33/67% quantiles computed on TRAIN samples only.",
              "- Environment inference is untouched; semantic labels are never model inputs.",
              "- Significance uses 1000 fixed-margin random contingency tables, mathematically equivalent to permuting environment labels.",
              "", "Files per case", "- sample_environment_assignments.csv: raw environment IDs and q.",
              "- semantic_labels.csv: aligned post-hoc labels and raw-window summaries.",
              "- environment_composition.csv: both conditional-probability directions, explicitly named.",
              "- environment_enrichment.csv and semantic_soft_assignment.csv: plot source values.",
              "- environment_semantic_metrics.json/txt: statistics and full provenance.", "",
              f"Completed cases: {len(provenance)}"]
    (output_root / "README.txt").write_text("\n".join(readme) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_root", type=Path, default=Path("results/fpem_environment_semantic_analysis"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS), choices=HORIZONS)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main():
    args = parse_args(); repo = args.repo.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    records = [r for r in discover_best_configs(repo / "results")
               if r["dataset"] in args.datasets and r["pred_len"] in args.horizons]
    expected = {(d, h) for d in args.datasets for h in args.horizons}
    found = {(r["dataset"], r["pred_len"]) for r in records}
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"missing exact best-run final environments: {missing}")
    all_metrics, all_enrichment, provenance = [], [], []
    priority = {"Weather": 0, "ETTm1": 1, "ETTh1": 2, "ETTh2": 3, "ETTm2": 4, "ExchangeRate": 5}
    for record in sorted(records, key=lambda r: (priority[r["dataset"]], r["pred_len"])):
        print(f"[semantic] {record['dataset']} pred={record['pred_len']} <- {record['environment_file']}", flush=True)
        metrics, enrichment, source = analyze_case(repo, record, output_root, args.permutations, args.seed)
        all_metrics.extend(metrics); all_enrichment.extend(enrichment); provenance.append(source)
    summarize(output_root, all_metrics, all_enrichment, provenance)
    print(f"[semantic] complete: {output_root}")


if __name__ == "__main__":
    main()
