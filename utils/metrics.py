import os

import numpy as np


METRIC_NAMES = ("MAE", "MSE", "RMSE", "MAPE", "MSPE")


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def _metric_values(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < len(METRIC_NAMES):
        raise ValueError("metrics must contain MAE, MSE, RMSE, MAPE and MSPE")
    return array[: len(METRIC_NAMES)]


def rebuild_readable_metrics(results_dir):
    """Rebuild one tab-separated text summary from all metrics.npy files."""
    results_dir = os.path.abspath(results_dir)
    rows = []
    if not os.path.isdir(results_dir):
        return None
    for experiment in sorted(os.listdir(results_dir)):
        metrics_path = os.path.join(results_dir, experiment, "metrics.npy")
        if not os.path.isfile(metrics_path):
            continue
        values = _metric_values(np.load(metrics_path, allow_pickle=False))
        rows.append((experiment, values))
    summary_path = os.path.join(results_dir, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("experiment\t" + "\t".join(METRIC_NAMES) + "\n")
        for experiment, values in rows:
            formatted = "\t".join("{:.10g}".format(float(value)) for value in values)
            handle.write("{}\t{}\n".format(experiment, formatted))
    return summary_path


def save_readable_metrics(folder_path, setting, values, dtw="Not calculated"):
    """Write human-readable metrics beside metrics.npy and refresh the index."""
    values = _metric_values(values)
    os.makedirs(folder_path, exist_ok=True)
    text_path = os.path.join(folder_path, "metrics.txt")
    with open(text_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("Experiment: {}\n".format(setting))
        for name, value in zip(METRIC_NAMES, values):
            handle.write("{}: {:.10g}\n".format(name, float(value)))
        handle.write("DTW: {}\n".format(dtw))
        handle.write("metrics.npy order: MAE, MSE, RMSE, MAPE, MSPE\n")
    rebuild_readable_metrics(os.path.dirname(os.path.abspath(folder_path.rstrip(os.sep))))
    return text_path


def backfill_readable_metrics(results_dir):
    """Create metrics.txt for historical result folders without rerunning models."""
    results_dir = os.path.abspath(results_dir)
    written = []
    if not os.path.isdir(results_dir):
        return written
    for experiment in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, experiment)
        metrics_path = os.path.join(folder_path, "metrics.npy")
        if not os.path.isfile(metrics_path):
            continue
        values = np.load(metrics_path, allow_pickle=False)
        text_path = os.path.join(folder_path, "metrics.txt")
        parsed = _metric_values(values)
        with open(text_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("Experiment: {}\n".format(experiment))
            for name, value in zip(METRIC_NAMES, parsed):
                handle.write("{}: {:.10g}\n".format(name, float(value)))
            handle.write("DTW: Not available in historical metrics.npy\n")
            handle.write("metrics.npy order: MAE, MSE, RMSE, MAPE, MSPE\n")
        written.append(text_path)
    rebuild_readable_metrics(results_dir)
    return written
