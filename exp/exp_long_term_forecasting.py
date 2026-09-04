from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, save_readable_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import glob
import time
import warnings
import numpy as np
from torch.utils.data import DataLoader
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


def _model_unwrapped(model):
    return model.module if hasattr(model, "module") else model


def _fpem_extra_loss(model, target):
    module = _model_unwrapped(model)
    if hasattr(module, "fpem_extra_loss"):
        return module.fpem_extra_loss(target)
    return target.new_zeros(()), {}


def _accumulate_logs(total, values):
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value)


def _rank_values(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _safe_correlation(left, right, spearman=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if spearman:
        left, right = _rank_values(left), _rank_values(right)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        model_optim = optim.Adam(trainable, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _prepare_fpem_pmg_graph(self, train_data, path, force=False):
        """Construct the graph with a chronological TRAIN-only loader.

        Validation and test datasets are deliberately not accepted by this
        method, which makes graph construction structurally leakage-safe.
        """
        module = _model_unwrapped(self.model)
        if not hasattr(module, "prepare_fpem_pmg_graph"):
            return None
        graph_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        metadata = {
            "dataset": self.args.data,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,
            "stage0_protocol": getattr(module, "stage0_protocol", ""),
            "representation_space": getattr(module, "representation_space", "hidden"),
            "patch_len": getattr(module, "patch_len", self.args.patch_len),
            "stride": getattr(module, "stride", 8),
            "d_model": self.args.d_model,
            "pattern_dim": getattr(getattr(module, "pmg", None), "pattern_dim", None),
            "projector_used": getattr(module, "representation_space", "embedding") != "raw",
            "predictive_stability_mode": getattr(module, "predictive_stability_mode", "per_future_znorm"),
        }
        return module.prepare_fpem_pmg_graph(graph_loader, self.device, path, metadata, force=force)

    def _fpem_pmg_warmup_loss(self, data_loader):
        module = _model_unwrapped(self.model)
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _ in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = module.fpem_pmg_warmup_forecast(batch_x)
                else:
                    outputs = module.fpem_pmg_warmup_forecast(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                loss = F.mse_loss(
                    outputs[:, -self.args.pred_len:, f_dim:],
                    batch_y[:, -self.args.pred_len:, f_dim:],
                )
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def _warmup_fpem_pmg_backbone(self, train_loader, vali_loader, test_loader, path):
        """Full forecasting warmup selected by validation early stopping."""
        module = _model_unwrapped(self.model)
        if not hasattr(module, "fpem_pmg_graph_available"):
            return None
        if module.fpem_pmg_graph_available(path):
            return None
        if str(getattr(self.args, "fpem_pmg_resume_checkpoint", "") or ""):
            return None
        a0_checkpoint = str(getattr(self.args, "fpem_pmg_a0_checkpoint", "") or "")
        loaded_a0 = False
        if a0_checkpoint and os.path.isfile(a0_checkpoint):
            try:
                loaded = module.load_fpem_pmg_a0_checkpoint(a0_checkpoint)
                loaded_a0 = True
                print("FPem-PMG Stage 0 loaded {} compatible A0 backbone/head tensors from {}".format(
                    loaded, a0_checkpoint
                ))
            except (KeyError, RuntimeError, ValueError) as error:
                print("FPem-PMG Stage 0 could not reuse A0 checkpoint; full warmup starts from scratch: {}".format(error))
        parameters = list(module.encoder_backbone.parameters()) + list(module.head_shared.parameters())
        if getattr(module, "representation_space", "embedding") != "raw":
            parameters += list(module.pmg.projector.parameters())
        optimizer = optim.Adam(parameters, lr=self.args.learning_rate)
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        checkpoint_path = os.path.join(path, "stage0_checkpoint.pth")
        best_val = float("inf")
        best_record = None
        stale_epochs = 0
        for epoch in range(self.args.train_epochs):
            losses = []
            self.model.train()
            for batch_x, batch_y, _, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                optimizer.zero_grad()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = module.fpem_pmg_warmup_forecast(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        loss = F.mse_loss(
                            outputs[:, -self.args.pred_len:, f_dim:],
                            batch_y[:, -self.args.pred_len:, f_dim:],
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = module.fpem_pmg_warmup_forecast(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    loss = F.mse_loss(
                        outputs[:, -self.args.pred_len:, f_dim:],
                        batch_y[:, -self.args.pred_len:, f_dim:],
                    )
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses))
            val_loss = self._fpem_pmg_warmup_loss(vali_loader)
            test_loss = self._fpem_pmg_warmup_loss(test_loader)
            projector_grad_terms = [
                parameter.grad.detach().float().square().sum()
                for parameter in module.pmg.projector.parameters()
                if parameter.grad is not None
            ]
            projector_grad_norm = (
                torch.sqrt(torch.stack(projector_grad_terms).sum()).item()
                if projector_grad_terms else 0.0
            )
            print(
                "FPem-PMG Stage 0 Epoch: {} | Train: {:.7f} Val: {:.7f} Test: {:.7f}".format(
                    epoch + 1, train_loss, val_loss, test_loss
                )
            )
            if val_loss < best_val:
                best_val = val_loss
                stale_epochs = 0
                best_record = {
                    "best_epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "loaded_a0": loaded_a0,
                    "a0_checkpoint": a0_checkpoint if loaded_a0 else "",
                    "projector_grad_norm": projector_grad_norm,
                }
                torch.save(
                    {
                        "encoder_backbone": module.encoder_backbone.state_dict(),
                        "pattern_projector": module.pmg.projector.state_dict(),
                        "head_shared": module.head_shared.state_dict(),
                        "record": best_record,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= self.args.patience:
                    print("FPem-PMG Stage 0 early stopping")
                    break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        saved = torch.load(checkpoint_path, map_location="cpu")
        module.encoder_backbone.load_state_dict(saved["encoder_backbone"])
        module.pmg.projector.load_state_dict(saved["pattern_projector"])
        module.head_shared.load_state_dict(saved["head_shared"])
        best_record = dict(saved["record"])
        with torch.no_grad():
            projector_norm = torch.sqrt(sum(
                parameter.detach().float().square().sum()
                for parameter in module.pmg.projector.parameters()
            )).item()
            representation_sum = None
            representation_sum_sq = None
            representation_count = 0
            self.model.eval()
            for batch_x, _, _, _ in train_loader:
                query = module.fpem_pmg_pattern_representation(batch_x.float().to(self.device))
                flat = query.float().reshape(-1, query.shape[-1])
                batch_sum = flat.sum(0).cpu()
                batch_sum_sq = flat.square().sum(0).cpu()
                representation_sum = batch_sum if representation_sum is None else representation_sum + batch_sum
                representation_sum_sq = batch_sum_sq if representation_sum_sq is None else representation_sum_sq + batch_sum_sq
                representation_count += flat.shape[0]
            representation_mean = representation_sum / float(representation_count)
            representation_variance = (
                representation_sum_sq / float(representation_count) - representation_mean.square()
            ).clamp_min(0.0).mean().item()
        best_record["projector_parameter_norm"] = projector_norm
        best_record["pattern_representation_variance"] = representation_variance
        report_path = os.path.join(path, "stage0_warmup.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            for key in (
                "loaded_a0", "a0_checkpoint", "best_epoch", "train_loss", "val_loss",
                "test_loss", "projector_parameter_norm", "pattern_representation_variance",
                "projector_grad_norm",
            ):
                handle.write("{}: {}\n".format(key, best_record[key]))
        print(
            "FPem-PMG Stage 0 best epoch={} train={:.7f} val={:.7f} test={:.7f} "
            "projector_grad_norm={:.7f} projector_norm={:.7f} q_variance={:.7g}".format(
                best_record["best_epoch"], best_record["train_loss"], best_record["val_loss"],
                best_record["test_loss"], best_record["projector_grad_norm"], projector_norm,
                representation_variance,
            )
        )
        return best_record

    def _calibrate_fpem_pmg_reliability(self, train_data, path):
        module = _model_unwrapped(self.model)
        if not hasattr(module, "calibrate_fpem_pmg_reliability"):
            return None
        train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        return module.calibrate_fpem_pmg_reliability(
            train_loader,
            self.device,
            path,
            target_channel_start=f_dim,
        )
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._warmup_fpem_pmg_backbone(train_loader, vali_loader, test_loader, path)
        self._prepare_fpem_pmg_graph(train_data, path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        start_epoch = 0
        resume_checkpoint = str(getattr(self.args, "fpem_pmg_resume_checkpoint", "") or "")
        if resume_checkpoint:
            resume_state = torch.load(resume_checkpoint, map_location="cpu")
            if isinstance(resume_state, dict) and "model" in resume_state:
                self.model.load_state_dict(resume_state["model"])
                if "optimizer" in resume_state:
                    model_optim.load_state_dict(resume_state["optimizer"])
                start_epoch = int(resume_state.get("epoch", -1)) + 1
            else:
                self.model.load_state_dict(resume_state)
            # Re-assert the graph's saved representation coordinates after a
            # checkpoint load so Stage 2 cannot query it in another space.
            self._prepare_fpem_pmg_graph(train_data, path)
            print("Resumed FPem-PMG training from {} at epoch {}".format(resume_checkpoint, start_epoch))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_fpem_logs = {}
            epoch_fpem_steps = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        extra_loss, _extra_logs = _fpem_extra_loss(self.model, batch_y)
                        loss = loss + extra_loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    extra_loss, _extra_logs = _fpem_extra_loss(self.model, batch_y)
                    loss = loss + extra_loss
                    train_loss.append(loss.item())

                if _extra_logs:
                    _accumulate_logs(epoch_fpem_logs, _extra_logs)
                    epoch_fpem_steps += 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if epoch_fpem_steps:
                averaged = {key: value / epoch_fpem_steps for key, value in epoch_fpem_logs.items()}
                print("FPem diagnostics: " + " | ".join("{}={:.6g}".format(key, value) for key, value in sorted(averaged.items())))
                with open(os.path.join(path, "stage2_diagnostics.txt"), "a", encoding="utf-8") as handle:
                    handle.write("epoch={}\n".format(epoch + 1))
                    for key, value in sorted(averaged.items()):
                        handle.write("{}: {:.10g}\n".format(key, value))
                    handle.write("\n")
            early_stopping(vali_loss, self.model, path)
            module = _model_unwrapped(self.model)
            if bool(getattr(module, "fpem_pmg_enabled", False)):
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": model_optim.state_dict(),
                        "epoch": epoch,
                        "graph_metadata": dict(getattr(module, "_graph_metadata", {})),
                    },
                    os.path.join(path, "training_state.pth"),
                )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            refresh_every = int(getattr(self.args, "fpem_pmg_refresh_every", 0))
            if refresh_every > 0 and (epoch + 1) % refresh_every == 0:
                self._prepare_fpem_pmg_graph(train_data, path, force=True)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        self._calibrate_fpem_pmg_reliability(train_data, path)
        torch.save(self.model.state_dict(), best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cpu"))

        preds = []
        trues = []
        module = _model_unwrapped(self.model)
        is_pmg = bool(getattr(module, "fpem_pmg_enabled", False))
        is_a1 = str(getattr(self.args, "fpem_pmg_ablation", "")).upper() == "A1"
        diagnostic_values = {
            "c_inv": [], "c_pat": [], "c_map": [],
            "pattern_novelty": [], "mapping_novelty": [], "null_probability": [],
            "variation_activation": [], "prediction_inv": [], "prediction_env": [],
            "prediction_shape": [], "prediction_scale": [],
            "prediction_shift": [], "prediction_typed": [],
            "graph_source_variance": [], "graph_source_norm": [],
            "query_variance": [], "query_norm": [],
            "graph_gate": [], "correction_norm": [], "context_norm": [],
            "invariant_base_norm": [], "zinv_norm": [], "correction_ratio": [],
            "stable_delta_norm": [], "mapping_deviation_norm": [],
            "graph_gate_tokens": [], "correction_norm_tokens": [],
            "correction_ratio_tokens": [], "stable_delta_norm_tokens": [],
            "mapping_deviation_norm_tokens": [],
            "raw_pattern_distance": [], "raw_deviation_norm": [],
            "raw_reconstruction_error": [], "raw_deviation_norm_tokens": [],
            "raw_shape_variation": [], "raw_shift_signed": [], "raw_scale_signed": [],
            "raw_abs_shift": [], "raw_abs_scale": [], "raw_u_shift": [], "raw_u_scale": [],
            "raw_a_shape": [], "raw_a_geo": [], "raw_a_var": [],
            "raw_stable_level": [], "raw_current_level": [],
            "raw_stable_scale": [], "raw_current_scale": [],
            "raw_gate": [], "raw_correction_norm": [], "latent_variant_norm": [],
            "raw_stable_reconstruction_norm": [],
            "typed_shape_gate": [], "typed_scale_gate": [], "typed_shift_gate": [],
            "typed_shape_gate_tokens": [], "typed_scale_gate_tokens": [], "typed_shift_gate_tokens": [],
            "typed_shape_correction_norm": [], "typed_shift_bias_norm": [],
            "typed_log_scale_mod": [], "typed_log_scale_mod_abs": [],
            "typed_scale_factor_tokens": [], "typed_z_inv_norm": [],
            "typed_z_shape_norm": [], "typed_z_scale_norm": [], "typed_z_final_norm": [],
            "c_shape": [], "c_rec": [], "c_pred": [],
            "zvar_norm": [], "zvar_norm_tokens": [],
            "projected_raw_norm": [], "projected_raw_norm_tokens": [],
            "hidden_norm": [], "hidden_norm_tokens": [],
            "mapping_pattern_error": [], "mapping_conditioned_error": [],
            "relation_variant_activation": [], "relation_p_var": [],
            "relation_active_edges": [], "relation_gate": [], "relation_correction_norm": [],
            "prediction_without_var_map": [], "prediction_with_var_map": [],
            "future_retrieval_confidence": [], "future_effective_entries": [],
            "future_memory_distance": [], "future_retrieved_p_inv": [],
            "future_variant_gate": [], "future_delta_logits_norm": [],
            "future_stable_logits_norm": [], "future_final_logits_norm": [],
            "future_delta_logits_flat": [], "future_ce_stable": [], "future_ce_full": [],
            "prediction_before_future_mapping": [],
        }
        baseline_model = None
        baseline_preds = []
        shuffled_mapping_preds = []
        a0_checkpoint = str(getattr(self.args, "fpem_pmg_a0_checkpoint", "") or "")
        if is_a1 and a0_checkpoint and os.path.isfile(a0_checkpoint):
            baseline_model = self.model_dict["PatchTST"](self.args).float().to(self.device)
            baseline_model.load_state_dict(torch.load(a0_checkpoint, map_location="cpu"))
            baseline_model.eval()
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                prediction_inv = None
                prediction_env = None
                prediction_shape = None
                prediction_scale = None
                prediction_shift = None
                prediction_typed = None
                prediction_without_var_map = None
                prediction_with_var_map = None
                if is_pmg:
                    latest = module.latest_fpem
                    if getattr(module, "future_mapping_mode", "off") != "off":
                        beta = module.fpem_future_target_responsibility(batch_y[:, -self.args.pred_len:, :])
                        stable_ce = -(beta * torch.log_softmax(latest["stable_future_logits"], -1)).sum(-1).mean((1, 2))
                        full_ce = -(beta * torch.log_softmax(latest["final_future_logits"], -1)).sum(-1).mean((1, 2))
                        future_tensors = {
                            "future_retrieval_confidence": latest["retrieval_confidence"],
                            "future_effective_entries": latest["effective_memory_entries"],
                            "future_memory_distance": latest["history_memory_distance"],
                            "future_retrieved_p_inv": latest["retrieved_p_inv"],
                            "future_variant_gate": latest["future_variant_gate"].squeeze(-1).squeeze(-1),
                            "future_delta_logits_norm": latest["variant_future_logits"].norm(dim=-1).mean(2),
                            "future_stable_logits_norm": latest["stable_future_logits"].norm(dim=-1).mean(2),
                            "future_final_logits_norm": latest["final_future_logits"].norm(dim=-1).mean(2),
                        }
                        for name, tensor in future_tensors.items():
                            diagnostic_values[name].append(tensor.mean(1).detach().cpu().numpy())
                        diagnostic_values["future_delta_logits_flat"].append(
                            latest["variant_future_logits"].detach().cpu().numpy().reshape(latest["variant_future_logits"].shape[0], -1)
                        )
                        diagnostic_values["future_ce_stable"].append(stable_ce.detach().cpu().numpy())
                        diagnostic_values["future_ce_full"].append(full_ce.detach().cpu().numpy())
                    diagnostic_values["c_inv"].append(latest["c_inv"].mean(dim=(1, 2)).detach().cpu().numpy())
                    diagnostic_values["c_pat"].append(latest["c_pat"].mean(dim=(1, 2)).detach().cpu().numpy())
                    diagnostic_values["c_map"].append(latest["c_map"].mean(dim=(1, 2)).detach().cpu().numpy())
                    diagnostic_values["pattern_novelty"].append(
                        latest["pattern_novelty"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["mapping_novelty"].append(
                        latest["mapping_novelty"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["null_probability"].append(
                        latest["null_probability"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["variation_activation"].append(
                        latest["variation_activation"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    relation_correction_norm = latest["variant_mapping_correction"].float().norm(dim=-1)
                    for name, tensor in (
                        ("relation_variant_activation", latest["variant_mapping_activation"]),
                        ("relation_p_var", latest["variant_mapping_probability"]),
                        ("relation_active_edges", latest["active_variant_edges"]),
                        ("relation_gate", latest["variant_mapping_gate"].squeeze(-1)),
                        ("relation_correction_norm", relation_correction_norm),
                    ):
                        diagnostic_values[name].append(tensor.mean(dim=(1, 2)).detach().cpu().numpy())
                    diagnostic_values["c_shape"].append(
                        latest["c_shape"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["c_rec"].append(
                        latest["c_rec"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["c_pred"].append(
                        latest["c_pred"].mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["graph_source_variance"].append(
                        float(latest["graph_tokens"].float().var(unbiased=False).detach().cpu())
                    )
                    diagnostic_values["graph_source_norm"].append(
                        float(latest["graph_tokens"].float().norm(dim=-1).mean().detach().cpu())
                    )
                    diagnostic_values["query_variance"].append(
                        float(latest["query"].float().var(unbiased=False).detach().cpu())
                    )
                    diagnostic_values["query_norm"].append(
                        float(latest["query"].float().norm(dim=-1).mean().detach().cpu())
                    )
                    correction_norm = latest["graph_correction"].float().norm(dim=-1)
                    invariant_base_norm = latest["invariant_base"].float().norm(dim=-1)
                    diagnostic_values["graph_gate"].append(
                        latest["graph_gate"].mean(dim=(1, 2, 3)).detach().cpu().numpy()
                    )
                    diagnostic_values["graph_gate_tokens"].append(
                        latest["graph_gate"].detach().cpu().numpy().reshape(-1)
                    )
                    diagnostic_values["correction_norm"].append(
                        correction_norm.mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["correction_norm_tokens"].append(
                        correction_norm.detach().cpu().numpy().reshape(-1)
                    )
                    diagnostic_values["context_norm"].append(
                        latest["consensus_context"].float().norm(dim=-1).mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["invariant_base_norm"].append(
                        invariant_base_norm.mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["zinv_norm"].append(
                        latest["z_inv"].float().norm(dim=-1).mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["correction_ratio"].append(
                        (correction_norm / invariant_base_norm.clamp_min(1e-6))
                        .mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["correction_ratio_tokens"].append(
                        (correction_norm / invariant_base_norm.clamp_min(1e-6))
                        .detach().cpu().numpy().reshape(-1)
                    )
                    diagnostic_values["stable_delta_norm"].append(
                        latest["stable_mapping_delta"].float().norm(dim=-1)
                        .mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["stable_delta_norm_tokens"].append(
                        latest["stable_mapping_delta"].float().norm(dim=-1)
                        .detach().cpu().numpy().reshape(-1)
                    )
                    diagnostic_values["mapping_deviation_norm"].append(
                        latest["mapping_deviation"].float().norm(dim=-1)
                        .mean(dim=(1, 2)).detach().cpu().numpy()
                    )
                    diagnostic_values["mapping_deviation_norm_tokens"].append(
                        latest["mapping_deviation"].float().norm(dim=-1)
                        .detach().cpu().numpy().reshape(-1)
                    )
                    zvar_norm = latest["z_var"].float().norm(dim=-1)
                    projected_raw_norm = latest["raw_deviation_projected"].float().norm(dim=-1)
                    hidden_norm = latest["hidden"].float().norm(dim=-1)
                    for sample_key, token_key, norm_values in (
                        ("zvar_norm", "zvar_norm_tokens", zvar_norm),
                        ("projected_raw_norm", "projected_raw_norm_tokens", projected_raw_norm),
                        ("hidden_norm", "hidden_norm_tokens", hidden_norm),
                    ):
                        diagnostic_values[sample_key].append(
                            norm_values.mean(dim=(1, 2)).detach().cpu().numpy()
                        )
                        diagnostic_values[token_key].append(
                            norm_values.detach().cpu().numpy().reshape(-1)
                        )
                    typed_token_values = {
                        "typed_shape_gate": latest["shape_gate"].squeeze(-1),
                        "typed_scale_gate": latest["scale_gate"].squeeze(-1),
                        "typed_shift_gate": latest["shift_gate"].squeeze(-1),
                        "typed_shape_correction_norm": latest["shape_correction"].float().norm(dim=-1),
                        "typed_shift_bias_norm": latest["shift_bias"].float().norm(dim=-1),
                        "typed_log_scale_mod": latest["log_scale_mod"].mean(dim=-1),
                        "typed_log_scale_mod_abs": latest["log_scale_mod"].abs().mean(dim=-1),
                        "typed_z_inv_norm": latest["z_inv"].float().norm(dim=-1),
                        "typed_z_shape_norm": latest["z_after_shape"].float().norm(dim=-1),
                        "typed_z_scale_norm": latest["z_after_scale"].float().norm(dim=-1),
                        "typed_z_final_norm": latest["z_typed"].float().norm(dim=-1),
                    }
                    for typed_name, typed_values in typed_token_values.items():
                        diagnostic_values[typed_name].append(
                            typed_values.mean(dim=(1, 2)).detach().cpu().numpy()
                        )
                    for typed_name in ("typed_shape_gate", "typed_scale_gate", "typed_shift_gate"):
                        diagnostic_values[typed_name + "_tokens"].append(
                            typed_token_values[typed_name].detach().cpu().numpy().reshape(-1)
                        )
                    diagnostic_values["typed_scale_factor_tokens"].append(
                        latest["scale_factor"].detach().cpu().numpy().reshape(-1)
                    )
                    if getattr(module, "representation_space", "embedding") == "raw":
                        diagnostic_values["raw_pattern_distance"].append(
                            latest["pattern_best_distance"].mean(dim=(1, 2)).detach().cpu().numpy()
                        )
                        diagnostic_values["raw_deviation_norm"].append(
                            latest["raw_deviation_norm"].mean(dim=(1, 2)).detach().cpu().numpy()
                        )
                        diagnostic_values["raw_reconstruction_error"].append(
                            latest["raw_reconstruction_error"].mean(dim=(1, 2)).detach().cpu().numpy()
                        )
                        diagnostic_values["raw_deviation_norm_tokens"].append(
                            latest["raw_deviation_norm"].detach().cpu().numpy().reshape(-1)
                        )
                        raw_shape_norm = latest["raw_variation_shape"].float().norm(dim=-1)
                        raw_correction_norm = latest["raw_variation_correction"].float().norm(dim=-1)
                        latent_variant_norm = latest["variant_latent"].float().norm(dim=-1)
                        raw_sample_values = {
                            "raw_shape_variation": raw_shape_norm,
                            "raw_shift_signed": latest["raw_variation_shift_signed"],
                            "raw_scale_signed": latest["raw_variation_scale_signed"],
                            "raw_abs_shift": latest["raw_variation_shift_signed"].abs(),
                            "raw_abs_scale": latest["raw_variation_scale_signed"].abs(),
                            "raw_u_shift": latest["raw_variation_u_shift"],
                            "raw_u_scale": latest["raw_variation_u_scale"],
                            "raw_a_shape": latest["raw_variation_a_shape"],
                            "raw_a_geo": latest["raw_variation_a_geo"],
                            "raw_a_var": latest["raw_variation_a_var"],
                            "raw_stable_level": latest["raw_stable_level"],
                            "raw_current_level": latest["raw_patch_mean"],
                            "raw_stable_scale": latest["raw_stable_scale"],
                            "raw_current_scale": latest["raw_patch_std"],
                            "raw_gate": latest["raw_variation_gate"].squeeze(-1),
                            "raw_correction_norm": raw_correction_norm,
                            "latent_variant_norm": latent_variant_norm,
                            "raw_stable_reconstruction_norm": latest["raw_stable_reconstruction_norm"],
                        }
                        for raw_name, raw_values in raw_sample_values.items():
                            diagnostic_values[raw_name].append(
                                raw_values.mean(dim=(1, 2)).detach().cpu().numpy()
                            )
                    prediction_inv = latest["prediction_inv"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_env = latest["prediction_env"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_shape = latest["prediction_shape"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_scale = latest["prediction_scale"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_shift = latest["prediction_shift"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_typed = latest["prediction_typed"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    if getattr(module, "future_mapping_mode", "off") != "off":
                        diagnostic_values["prediction_before_future_mapping"].append(
                            latest["prediction_before_future_mapping"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                        )
                    prediction_without_var_map = latest["prediction_without_var_map"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    prediction_with_var_map = latest["prediction_with_var_map"][:, -self.args.pred_len:, :].detach().cpu().numpy()
                    if latest["pattern_future_prediction"].shape[-1] > 0:
                        future_representation = module.fpem_pmg_future_representation(batch_y, batch_x)
                        target_future = future_representation[:, :, None, :]
                        pattern_future = latest["pattern_future_prediction"]
                        mapping_future = latest["mapping_future_prediction"]
                        patch_slice = slice(1, None) if pattern_future.shape[2] > 1 else slice(None)
                        diagnostic_values["mapping_pattern_error"].append(
                            (pattern_future[:, :, patch_slice, :] - target_future).square()
                            .mean(dim=(1, 2, 3)).detach().cpu().numpy()
                        )
                        diagnostic_values["mapping_conditioned_error"].append(
                            (mapping_future[:, :, patch_slice, :] - target_future).square()
                            .mean(dim=(1, 2, 3)).detach().cpu().numpy()
                        )
                    shuffled_mapping_output = module.fpem_pmg_shuffled_mapping_forecast(batch_x)
                    shuffled_mapping_output = shuffled_mapping_output[:, -self.args.pred_len:, :].detach().cpu().numpy()
                baseline_output = None
                if baseline_model is not None:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            baseline_output = baseline_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        baseline_output = baseline_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    baseline_output = baseline_output[:, -self.args.pred_len:, :].detach().cpu().numpy()
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    if prediction_inv is not None:
                        prediction_inv = test_data.inverse_transform(
                            prediction_inv.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_env = test_data.inverse_transform(
                            prediction_env.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_shape = test_data.inverse_transform(
                            prediction_shape.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_scale = test_data.inverse_transform(
                            prediction_scale.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_shift = test_data.inverse_transform(
                            prediction_shift.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_typed = test_data.inverse_transform(
                            prediction_typed.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_without_var_map = test_data.inverse_transform(
                            prediction_without_var_map.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        prediction_with_var_map = test_data.inverse_transform(
                            prediction_with_var_map.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    if baseline_output is not None:
                        baseline_output = test_data.inverse_transform(
                            baseline_output.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    if is_pmg:
                        shuffled_mapping_output = test_data.inverse_transform(
                            shuffled_mapping_output.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                if prediction_inv is not None:
                    diagnostic_values["prediction_inv"].append(prediction_inv[:, :, f_dim:])
                    diagnostic_values["prediction_env"].append(prediction_env[:, :, f_dim:])
                    diagnostic_values["prediction_shape"].append(prediction_shape[:, :, f_dim:])
                    diagnostic_values["prediction_scale"].append(prediction_scale[:, :, f_dim:])
                    diagnostic_values["prediction_shift"].append(prediction_shift[:, :, f_dim:])
                    diagnostic_values["prediction_typed"].append(prediction_typed[:, :, f_dim:])
                    diagnostic_values["prediction_without_var_map"].append(prediction_without_var_map[:, :, f_dim:])
                    diagnostic_values["prediction_with_var_map"].append(prediction_with_var_map[:, :, f_dim:])
                if baseline_output is not None:
                    baseline_preds.append(baseline_output[:, :, f_dim:])
                if is_pmg:
                    shuffled_mapping_preds.append(shuffled_mapping_output[:, :, f_dim:])

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        metric_values = np.array([mae, mse, rmse, mape, mspe])
        np.save(folder_path + 'metrics.npy', metric_values)
        save_readable_metrics(folder_path, setting, metric_values, dtw=dtw)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if is_pmg and diagnostic_values["c_inv"]:
            c_inv_samples = np.concatenate(diagnostic_values["c_inv"])
            c_pat_samples = np.concatenate(diagnostic_values["c_pat"])
            c_map_samples = np.concatenate(diagnostic_values["c_map"])
            pattern_novelty_samples = np.concatenate(diagnostic_values["pattern_novelty"])
            mapping_novelty_samples = np.concatenate(diagnostic_values["mapping_novelty"])
            null_samples = np.concatenate(diagnostic_values["null_probability"])
            activation_samples = np.concatenate(diagnostic_values["variation_activation"])
            graph_gate_samples = np.concatenate(diagnostic_values["graph_gate"])
            correction_norm_samples = np.concatenate(diagnostic_values["correction_norm"])
            context_norm_samples = np.concatenate(diagnostic_values["context_norm"])
            invariant_base_norm_samples = np.concatenate(diagnostic_values["invariant_base_norm"])
            zinv_norm_samples = np.concatenate(diagnostic_values["zinv_norm"])
            correction_ratio_samples = np.concatenate(diagnostic_values["correction_ratio"])
            stable_delta_norm_samples = np.concatenate(diagnostic_values["stable_delta_norm"])
            mapping_deviation_norm_samples = np.concatenate(diagnostic_values["mapping_deviation_norm"])
            graph_gate_tokens = np.concatenate(diagnostic_values["graph_gate_tokens"])
            correction_norm_tokens = np.concatenate(diagnostic_values["correction_norm_tokens"])
            correction_ratio_tokens = np.concatenate(diagnostic_values["correction_ratio_tokens"])
            stable_delta_norm_tokens = np.concatenate(diagnostic_values["stable_delta_norm_tokens"])
            mapping_deviation_norm_tokens = np.concatenate(diagnostic_values["mapping_deviation_norm_tokens"])
            c_shape_samples = np.concatenate(diagnostic_values["c_shape"])
            c_rec_samples = np.concatenate(diagnostic_values["c_rec"])
            c_pred_samples = np.concatenate(diagnostic_values["c_pred"])
            zvar_norm_samples = np.concatenate(diagnostic_values["zvar_norm"])
            zvar_norm_tokens = np.concatenate(diagnostic_values["zvar_norm_tokens"])
            projected_raw_norm_samples = np.concatenate(diagnostic_values["projected_raw_norm"])
            projected_raw_norm_tokens = np.concatenate(diagnostic_values["projected_raw_norm_tokens"])
            hidden_norm_tokens = np.concatenate(diagnostic_values["hidden_norm_tokens"])
            typed_samples = {
                name: np.concatenate(diagnostic_values[name])
                for name in (
                    "typed_shape_gate", "typed_scale_gate", "typed_shift_gate",
                    "typed_shape_correction_norm", "typed_shift_bias_norm",
                    "typed_log_scale_mod", "typed_log_scale_mod_abs",
                    "typed_z_inv_norm", "typed_z_shape_norm",
                    "typed_z_scale_norm", "typed_z_final_norm",
                )
            }
            typed_tokens = {
                name: np.concatenate(diagnostic_values[name])
                for name in (
                    "typed_shape_gate_tokens", "typed_scale_gate_tokens",
                    "typed_shift_gate_tokens", "typed_scale_factor_tokens",
                )
            }
            mapping_pattern_error = np.concatenate(diagnostic_values["mapping_pattern_error"])
            mapping_conditioned_error = np.concatenate(diagnostic_values["mapping_conditioned_error"])
            sample_mapping_gain = mapping_pattern_error - mapping_conditioned_error
            shuffled_mapping_prediction = np.concatenate(shuffled_mapping_preds)
            is_raw = getattr(module, "representation_space", "embedding") == "raw"
            if is_raw:
                raw_pattern_distance_samples = np.concatenate(diagnostic_values["raw_pattern_distance"])
                raw_deviation_norm_samples = np.concatenate(diagnostic_values["raw_deviation_norm"])
                raw_reconstruction_error_samples = np.concatenate(diagnostic_values["raw_reconstruction_error"])
                raw_deviation_norm_tokens = np.concatenate(diagnostic_values["raw_deviation_norm_tokens"])
                raw_component_samples = {
                    name: np.concatenate(diagnostic_values[name])
                    for name in (
                        "raw_shape_variation", "raw_shift_signed", "raw_scale_signed",
                        "raw_abs_shift", "raw_abs_scale", "raw_u_shift", "raw_u_scale",
                        "raw_a_shape", "raw_a_geo", "raw_a_var", "raw_stable_level",
                        "raw_current_level", "raw_stable_scale", "raw_current_scale",
                        "raw_gate", "raw_correction_norm", "latent_variant_norm",
                        "raw_stable_reconstruction_norm",
                    )
                }
            prediction_inv = np.concatenate(diagnostic_values["prediction_inv"])
            prediction_env = np.concatenate(diagnostic_values["prediction_env"])
            prediction_shape = np.concatenate(diagnostic_values["prediction_shape"])
            prediction_scale = np.concatenate(diagnostic_values["prediction_scale"])
            prediction_shift = np.concatenate(diagnostic_values["prediction_shift"])
            prediction_typed = np.concatenate(diagnostic_values["prediction_typed"])
            relation_samples = {
                name: np.concatenate(diagnostic_values[name]) for name in (
                    "relation_variant_activation", "relation_p_var", "relation_active_edges",
                    "relation_gate", "relation_correction_norm",
                )
            }
            prediction_without_var_map = np.concatenate(diagnostic_values["prediction_without_var_map"])
            prediction_with_var_map = np.concatenate(diagnostic_values["prediction_with_var_map"])
            relation_gain = (
                ((prediction_without_var_map - trues) ** 2).mean(axis=(1, 2))
                - ((prediction_with_var_map - trues) ** 2).mean(axis=(1, 2))
            )
            inv_mae, inv_mse, _, _, _ = metric(prediction_inv, trues)
            env_mse = float(np.mean((prediction_env - trues) ** 2))
            sample_inv_mse = ((prediction_inv - trues) ** 2).mean(axis=(1, 2))
            sample_env_mse = ((prediction_env - trues) ** 2).mean(axis=(1, 2))
            sample_env_gain = sample_inv_mse - sample_env_mse
            sample_shape_mse = ((prediction_shape - trues) ** 2).mean(axis=(1, 2))
            sample_scale_mse = ((prediction_scale - trues) ** 2).mean(axis=(1, 2))
            sample_shift_mse = ((prediction_shift - trues) ** 2).mean(axis=(1, 2))
            sample_typed_mse = ((prediction_typed - trues) ** 2).mean(axis=(1, 2))
            typed_gains = {
                "shape": sample_inv_mse - sample_shape_mse,
                "scale": sample_inv_mse - sample_scale_mse,
                "shift": sample_inv_mse - sample_shift_mse,
                "full": sample_inv_mse - sample_typed_mse,
            }
            pattern_support = module.pmg.pattern_graph.window_support.float()
            mapping_support = module.pmg.mapping_graph.window_support.float()
            diagnostics_path = os.path.join(folder_path, "fpem_diagnostics.txt")
            with open(diagnostics_path, "w", encoding="utf-8") as handle:
                handle.write("experiment: {}\n".format(setting))
                handle.write("representation_space: {}\n".format(module.representation_space))
                handle.write("graph/representation_space: {}\n".format(module.representation_space))
                handle.write("graph/projector_used: {}\n".format(str(not is_raw).lower()))
                handle.write("cinv/mode: {}\n".format(module.cinv_mode))
                handle.write("zinv/mode: {}\n".format(module.zinv_mode))
                handle.write("variant/input_mode: {}\n".format(module.variant_input_mode))
                handle.write("predictive_stability/mode: {}\n".format(module.predictive_stability_mode))
                handle.write("mapping/use_mode: {}\n".format(module.mapping_use_mode))
                handle.write("relation_mapping/mode: {}\n".format(module.relation_mapping_mode))
                handle.write("relation_mapping/source_split: train\n")
                handle.write("history_future/mode: {}\n".format(module.future_mapping_mode))
                handle.write("history_future/graph_source_split: train\n")
                handle.write("history_future/test_inference_uses_target: false\n")
                handle.write("fusion/mode: {}\n".format(module.fusion_mode))
                handle.write("fusion/typed_components: {}\n".format(module.typed_fusion_components))
                handle.write("stage2/freeze_patch_embedding: {}\n".format(str(module._patch_embedding_frozen).lower()))
                handle.write("zinv/initial_difference_from_hidden: {:.10g}\n".format(
                    float(module.zinv_initial_difference.detach().cpu())
                ))
                handle.write("graph_source/variance: {:.10g}\n".format(
                    float(np.mean(diagnostic_values["graph_source_variance"]))
                ))
                handle.write("graph_source/norm: {:.10g}\n".format(
                    float(np.mean(diagnostic_values["graph_source_norm"]))
                ))
                handle.write("query/variance: {:.10g}\n".format(
                    float(np.mean(diagnostic_values["query_variance"]))
                ))
                handle.write("query/norm: {:.10g}\n".format(
                    float(np.mean(diagnostic_values["query_norm"]))
                ))
                handle.write("pattern/num_active: {}\n".format(module.pmg.pattern_graph.num_active))
                handle.write("graph/num_patterns: {}\n".format(module.pmg.pattern_graph.num_active))
                handle.write("pattern/mean_support: {:.10g}\n".format(
                    float(pattern_support.mean().cpu()) if pattern_support.numel() else 0.0
                ))
                handle.write("pattern/mean_novelty: {:.10g}\n".format(float(pattern_novelty_samples.mean())))
                handle.write("pattern/novelty_mean: {:.10g}\n".format(float(pattern_novelty_samples.mean())))
                handle.write("pattern/null_rate: {:.10g}\n".format(float(null_samples.mean())))
                handle.write("mapping/num_active_edges: {}\n".format(module.pmg.mapping_graph.num_active))
                handle.write("graph/num_mappings: {}\n".format(module.pmg.mapping_graph.num_active))
                handle.write("mapping/mean_support: {:.10g}\n".format(
                    float(mapping_support.mean().cpu()) if mapping_support.numel() else 0.0
                ))
                handle.write("mapping/mean_novelty: {:.10g}\n".format(
                    float(mapping_novelty_samples.mean())
                ))
                handle.write("mapping/novelty_mean: {:.10g}\n".format(float(mapping_novelty_samples.mean())))
                mapping_graph = module.pmg.mapping_graph
                p_inv = mapping_graph.p_inv.detach().cpu().numpy()
                p_var = mapping_graph.p_var.detach().cpu().numpy()
                coverage = mapping_graph.coverage.detach().cpu().numpy()
                entropy = mapping_graph.sample_entropy.detach().cpu().numpy()
                concentration = mapping_graph.sample_concentration.detach().cpu().numpy()
                handle.write("relation/total_edges: {}\n".format(mapping_graph.num_active))
                for name, values_array in (("p_inv", p_inv), ("p_var", p_var)):
                    if values_array.size:
                        handle.write("relation/{}_mean: {:.10g}\n".format(name, float(values_array.mean())))
                        handle.write("relation/{}_median: {:.10g}\n".format(name, float(np.median(values_array))))
                        for qname, qvalue in (("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9)):
                            handle.write("relation/{}_{}: {:.10g}\n".format(name, qname, float(np.quantile(values_array, qvalue))))
                for feature_name, feature in (("coverage", coverage), ("entropy", entropy), ("concentration", concentration)):
                    if feature.size and p_inv.size:
                        handle.write("relation/{}_vs_p_inv_spearman: {:.10g}\n".format(feature_name, _safe_correlation(feature, p_inv, spearman=True)))
                        handle.write("relation/{}_vs_p_var_spearman: {:.10g}\n".format(feature_name, _safe_correlation(feature, p_var, spearman=True)))
                handle.write("relation/active_variant_edges_mean: {:.10g}\n".format(float(relation_samples["relation_active_edges"].mean())))
                handle.write("relation/variant_activation_mean: {:.10g}\n".format(float(relation_samples["relation_variant_activation"].mean())))
                handle.write("relation/variant_activation_sample_std: {:.10g}\n".format(float(relation_samples["relation_variant_activation"].std())))
                handle.write("relation/gate_mean: {:.10g}\n".format(float(relation_samples["relation_gate"].mean())))
                handle.write("relation/gate_std: {:.10g}\n".format(float(relation_samples["relation_gate"].std())))
                for qname, qvalue in (("p10", .1), ("p50", .5), ("p90", .9)):
                    handle.write("relation/gate_{}: {:.10g}\n".format(qname, float(np.quantile(relation_samples["relation_gate"], qvalue))))
                handle.write("relation/correction_norm_mean: {:.10g}\n".format(float(relation_samples["relation_correction_norm"].mean())))
                handle.write("relation/gain_mean: {:.10g}\n".format(float(relation_gain.mean())))
                handle.write("relation/gain_median: {:.10g}\n".format(float(np.median(relation_gain))))
                handle.write("relation/gain_positive_fraction: {:.10g}\n".format(float((relation_gain > 0).mean())))
                for feature_name in ("relation_p_var", "relation_variant_activation", "relation_gate"):
                    handle.write("relation/{}_vs_gain_spearman: {:.10g}\n".format(
                        feature_name, _safe_correlation(relation_samples[feature_name], relation_gain, spearman=True)
                    ))
                if module.future_mapping_mode != "off":
                    future_samples = {
                        name: np.concatenate(diagnostic_values[name]) for name in (
                            "future_retrieval_confidence", "future_effective_entries",
                            "future_memory_distance", "future_retrieved_p_inv",
                            "future_variant_gate", "future_delta_logits_norm",
                            "future_stable_logits_norm", "future_final_logits_norm",
                            "future_ce_stable", "future_ce_full",
                        )
                    }
                    future_gain = future_samples["future_ce_stable"] - future_samples["future_ce_full"]
                    delta_flat = np.concatenate(diagnostic_values["future_delta_logits_flat"])
                    memory = module.future_mapping_memory
                    for name, value in (
                        ("retrieval_confidence", future_samples["future_retrieval_confidence"]),
                        ("effective_memory_entries", future_samples["future_effective_entries"]),
                        ("history_memory_distance", future_samples["future_memory_distance"]),
                        ("retrieved_p_inv", future_samples["future_retrieved_p_inv"]),
                        ("variant_gate", future_samples["future_variant_gate"]),
                        ("delta_logits_norm", future_samples["future_delta_logits_norm"]),
                        ("stable_logits_norm", future_samples["future_stable_logits_norm"]),
                        ("final_logits_norm", future_samples["future_final_logits_norm"]),
                    ):
                        handle.write("history_future/{}_mean: {:.10g}\n".format(name, float(value.mean())))
                    handle.write("history_future/variant_gate_std: {:.10g}\n".format(float(future_samples["future_variant_gate"].std())))
                    for qname, qvalue in (("p10", .1), ("p50", .5), ("p90", .9)):
                        handle.write("history_future/variant_gate_{}: {:.10g}\n".format(qname, float(np.quantile(future_samples["future_variant_gate"], qvalue))))
                    handle.write("history_future/delta_logits_sample_diversity: {:.10g}\n".format(float(delta_flat.std(axis=0).mean())))
                    handle.write("history_future/ce_stable: {:.10g}\n".format(float(future_samples["future_ce_stable"].mean())))
                    handle.write("history_future/ce_full: {:.10g}\n".format(float(future_samples["future_ce_full"].mean())))
                    handle.write("history_future/variant_gain_mean: {:.10g}\n".format(float(future_gain.mean())))
                    handle.write("history_future/variant_gain_median: {:.10g}\n".format(float(np.median(future_gain))))
                    handle.write("history_future/positive_gain_fraction: {:.10g}\n".format(float((future_gain > 0).mean())))
                    handle.write("history_future/gate_vs_gain_spearman: {:.10g}\n".format(
                        _safe_correlation(future_samples["future_variant_gate"], future_gain, spearman=True)
                    ))
                    base_future_prediction = np.concatenate(diagnostic_values["prediction_before_future_mapping"])
                    base_future_mse = float(np.mean((base_future_prediction - trues) ** 2))
                    handle.write("history_future/base_forecast_mse: {:.10g}\n".format(base_future_mse))
                    handle.write("history_future/forecast_gain: {:.10g}\n".format(base_future_mse - float(mse)))
                    handle.write("history_future/memory_entries: {}\n".format(memory.history_keys.shape[0]))
                    handle.write("history_future/memory_p_inv_mean: {:.10g}\n".format(float(memory.p_inv.mean().cpu())))
                    handle.write("history_future/future_consistency_mean: {:.10g}\n".format(float(memory.future_consistency.mean().cpu())))
                handle.write("mapping/stable_delta_norm_mean: {:.10g}\n".format(
                    float(stable_delta_norm_tokens.mean())
                ))
                handle.write("mapping/stable_delta_norm_p90: {:.10g}\n".format(
                    float(np.quantile(stable_delta_norm_tokens, 0.9))
                ))
                handle.write("mapping/deviation_norm_mean: {:.10g}\n".format(
                    float(mapping_deviation_norm_tokens.mean())
                ))
                handle.write("mapping/deviation_norm_p90: {:.10g}\n".format(
                    float(np.quantile(mapping_deviation_norm_tokens, 0.9))
                ))
                handle.write("variant/activation_mean: {:.10g}\n".format(float(activation_samples.mean())))
                handle.write("variant/activation_p90: {:.10g}\n".format(float(np.quantile(activation_samples, 0.9))))
                handle.write("forecast/full_metric: {:.10g}\n".format(float(mse)))
                handle.write("forecast/inv_metric: {:.10g}\n".format(float(inv_mse)))
                handle.write("forecast/env_gain: {:.10g}\n".format(float(inv_mse - env_mse)))
                handle.write("c_inv_mean: {:.10g}\n".format(float(c_inv_samples.mean())))
                handle.write("c_inv_p10: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.1))))
                handle.write("c_inv_p50: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.5))))
                handle.write("c_inv_p90: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.9))))
                handle.write("cinv/mean: {:.10g}\n".format(float(c_inv_samples.mean())))
                handle.write("cinv/p10: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.1))))
                handle.write("cinv/p50: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.5))))
                handle.write("cinv/p90: {:.10g}\n".format(float(np.quantile(c_inv_samples, 0.9))))
                handle.write("cinv/std: {:.10g}\n".format(float(c_inv_samples.std())))
                handle.write("pattern/c_mean: {:.10g}\n".format(float(c_pat_samples.mean())))
                handle.write("pattern/c_shape_mean: {:.10g}\n".format(float(c_shape_samples.mean())))
                handle.write("pattern/c_rec_mean: {:.10g}\n".format(float(c_rec_samples.mean())))
                handle.write("pattern/c_pred_mean: {:.10g}\n".format(float(c_pred_samples.mean())))
                node_diagnostics = (
                    ("future_ratio", module.pmg.pattern_graph.future_ratio.detach().cpu().numpy()),
                    ("pattern_predictive_stability", module.pmg.pattern_graph.predictive_support.detach().cpu().numpy()),
                )
                for node_name, node_values in node_diagnostics:
                    if node_values.size:
                        for quantile_name, quantile in (
                            ("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
                            ("p75", 0.75), ("p90", 0.90), ("max", 1.0),
                        ):
                            handle.write("{}/{}: {:.10g}\n".format(
                                node_name, quantile_name, float(np.quantile(node_values, quantile))
                            ))
                        handle.write("{}/std: {:.10g}\n".format(node_name, float(node_values.std())))
                handle.write("mapping/c_mean: {:.10g}\n".format(float(c_map_samples.mean())))
                handle.write("pattern/c_std: {:.10g}\n".format(float(c_pat_samples.std())))
                handle.write("mapping/c_std: {:.10g}\n".format(float(c_map_samples.std())))
                if is_raw:
                    handle.write("raw/signal_scale: pre_patch_normalization_not_physical_raw_units\n")
                    handle.write("raw/pattern_distance_mean: {:.10g}\n".format(float(raw_pattern_distance_samples.mean())))
                    handle.write("raw/pattern_distance_p90: {:.10g}\n".format(float(np.quantile(raw_pattern_distance_samples, 0.9))))
                    handle.write("raw/reconstruction_error_mean: {:.10g}\n".format(float(raw_reconstruction_error_samples.mean())))
                    handle.write("raw/deviation_norm_mean: {:.10g}\n".format(float(raw_deviation_norm_tokens.mean())))
                    handle.write("raw_deviation/norm_mean: {:.10g}\n".format(float(raw_deviation_norm_tokens.mean())))
                    handle.write("raw_deviation/norm_p90: {:.10g}\n".format(float(np.quantile(raw_deviation_norm_tokens, 0.9))))
                    for output_name, sample_name in (
                        ("raw_variation/shape_norm", "raw_shape_variation"),
                        ("raw_variation/shift_signed", "raw_shift_signed"),
                        ("raw_variation/scale_signed", "raw_scale_signed"),
                        ("raw_variation/u_shift", "raw_u_shift"),
                        ("raw_variation/u_scale", "raw_u_scale"),
                        ("raw_variation/a_shape", "raw_a_shape"),
                        ("raw_variation/a_geo", "raw_a_geo"),
                        ("raw_variation/a_var", "raw_a_var"),
                        ("raw_variation/stable_level", "raw_stable_level"),
                        ("raw_variation/current_level", "raw_current_level"),
                        ("raw_variation/stable_scale", "raw_stable_scale"),
                        ("raw_variation/current_scale", "raw_current_scale"),
                        ("raw_variation/raw_gate", "raw_gate"),
                        ("raw_variation/raw_correction_norm", "raw_correction_norm"),
                        ("raw_variation/stable_reconstruction_norm", "raw_stable_reconstruction_norm"),
                        ("variant/latent_norm", "latent_variant_norm"),
                        ("variant/raw_correction_norm", "raw_correction_norm"),
                    ):
                        handle.write("{}: {:.10g}\n".format(
                            output_name, float(raw_component_samples[sample_name].mean())
                        ))
                    handle.write("variant/final_norm: {:.10g}\n".format(float(zvar_norm_samples.mean())))
                for init_name, init_value in (
                    ("z_diff", module.typed_initial_z_diff),
                    ("shape_correction_norm", module.typed_initial_shape_correction_norm),
                    ("scale_log_mod_norm", module.typed_initial_scale_log_mod_norm),
                    ("shift_bias_norm", module.typed_initial_shift_bias_norm),
                ):
                    handle.write("typed_init/{}: {:.10g}\n".format(
                        init_name, float(init_value.detach().cpu())
                    ))
                handle.write("typed/shape_variation_norm: {:.10g}\n".format(
                    float(raw_component_samples["raw_shape_variation"].mean()) if is_raw else 0.0
                ))
                for gate_name in ("shape", "scale", "shift"):
                    gate_values = typed_tokens["typed_{}_gate_tokens".format(gate_name)]
                    handle.write("typed/{}_gate_mean: {:.10g}\n".format(gate_name, float(gate_values.mean())))
                    handle.write("typed/{}_gate_p50: {:.10g}\n".format(gate_name, float(np.quantile(gate_values, 0.5))))
                    handle.write("typed/{}_gate_p90: {:.10g}\n".format(gate_name, float(np.quantile(gate_values, 0.9))))
                handle.write("typed/shape_correction_norm: {:.10g}\n".format(
                    float(typed_samples["typed_shape_correction_norm"].mean())
                ))
                handle.write("typed/scale_signed_abs_mean: {:.10g}\n".format(
                    float(raw_component_samples["raw_abs_scale"].mean()) if is_raw else 0.0
                ))
                handle.write("typed/u_scale_mean: {:.10g}\n".format(
                    float(raw_component_samples["raw_u_scale"].mean()) if is_raw else 0.0
                ))
                handle.write("typed/log_scale_mod_mean: {:.10g}\n".format(
                    float(typed_samples["typed_log_scale_mod"].mean())
                ))
                handle.write("typed/log_scale_mod_abs_mean: {:.10g}\n".format(
                    float(typed_samples["typed_log_scale_mod_abs"].mean())
                ))
                scale_factor_values = typed_tokens["typed_scale_factor_tokens"]
                handle.write("typed/scale_factor_mean: {:.10g}\n".format(float(scale_factor_values.mean())))
                for quantile_name, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
                    handle.write("typed/scale_factor_{}: {:.10g}\n".format(
                        quantile_name, float(np.quantile(scale_factor_values, quantile))
                    ))
                handle.write("typed/shift_signed_abs_mean: {:.10g}\n".format(
                    float(raw_component_samples["raw_abs_shift"].mean()) if is_raw else 0.0
                ))
                handle.write("typed/u_shift_mean: {:.10g}\n".format(
                    float(raw_component_samples["raw_u_shift"].mean()) if is_raw else 0.0
                ))
                handle.write("typed/shift_bias_norm: {:.10g}\n".format(
                    float(typed_samples["typed_shift_bias_norm"].mean())
                ))
                for output_name, sample_name in (
                    ("z_inv_norm", "typed_z_inv_norm"),
                    ("z_after_shape_norm", "typed_z_shape_norm"),
                    ("z_after_scale_norm", "typed_z_scale_norm"),
                    ("z_final_norm", "typed_z_final_norm"),
                ):
                    handle.write("typed/{}: {:.10g}\n".format(
                        output_name, float(typed_samples[sample_name].mean())
                    ))
                handle.write("typed/shape_correction_to_zinv_ratio: {:.10g}\n".format(float(
                    (typed_samples["typed_shape_correction_norm"] /
                     np.maximum(typed_samples["typed_z_inv_norm"], 1e-6)).mean()
                )))
                handle.write("typed/shift_bias_to_zinv_ratio: {:.10g}\n".format(float(
                    (typed_samples["typed_shift_bias_norm"] /
                     np.maximum(typed_samples["typed_z_inv_norm"], 1e-6)).mean()
                )))
                candidate_mse = {
                    "inv": sample_inv_mse.mean(), "shape": sample_shape_mse.mean(),
                    "scale": sample_scale_mse.mean(), "shift": sample_shift_mse.mean(),
                    "typed": sample_typed_mse.mean(),
                }
                for candidate_name, candidate_value in candidate_mse.items():
                    handle.write("typed_prediction/{}_MSE: {:.10g}\n".format(
                        candidate_name, float(candidate_value)
                    ))
                for gain_name, gain_values in typed_gains.items():
                    handle.write("typed_gain/{}_mean: {:.10g}\n".format(
                        gain_name, float(gain_values.mean())
                    ))
                    handle.write("typed_gain/{}_fraction_gain_positive: {:.10g}\n".format(
                        gain_name, float((gain_values > 0).mean())
                    ))
                handle.write("graph/correction_norm_mean: {:.10g}\n".format(float(correction_norm_tokens.mean())))
                handle.write("graph/correction_norm_p90: {:.10g}\n".format(float(np.quantile(correction_norm_tokens, 0.9))))
                handle.write("graph/gate_mean: {:.10g}\n".format(float(graph_gate_tokens.mean())))
                handle.write("graph/gate_p10: {:.10g}\n".format(float(np.quantile(graph_gate_tokens, 0.1))))
                handle.write("graph/gate_p50: {:.10g}\n".format(float(np.quantile(graph_gate_tokens, 0.5))))
                handle.write("graph/gate_p90: {:.10g}\n".format(float(np.quantile(graph_gate_tokens, 0.9))))
                handle.write("graph/context_norm_mean: {:.10g}\n".format(float(context_norm_samples.mean())))
                handle.write("hidden/invariant_base_norm_mean: {:.10g}\n".format(float(invariant_base_norm_samples.mean())))
                handle.write("zinv/norm_mean: {:.10g}\n".format(float(zinv_norm_samples.mean())))
                handle.write("graph/correction_to_hidden_ratio: {:.10g}\n".format(float(correction_ratio_tokens.mean())))
                handle.write("hidden/norm_mean: {:.10g}\n".format(float(hidden_norm_tokens.mean())))
                handle.write("zvar/norm_mean: {:.10g}\n".format(float(zvar_norm_tokens.mean())))
                handle.write("zvar/norm_p90: {:.10g}\n".format(float(np.quantile(zvar_norm_tokens, 0.9))))
                handle.write("raw_deviation_projected/norm_mean: {:.10g}\n".format(float(projected_raw_norm_tokens.mean())))
                handle.write("raw_deviation_projected/norm_p90: {:.10g}\n".format(float(np.quantile(projected_raw_norm_tokens, 0.9))))
                handle.write("raw_deviation_projected/to_hidden_ratio: {:.10g}\n".format(
                    float((projected_raw_norm_tokens / np.maximum(hidden_norm_tokens, 1e-6)).mean())
                ))
                conditional_pattern_mse = float(mapping_pattern_error.mean())
                conditional_mapping_mse = float(mapping_conditioned_error.mean())
                conditional_gain = conditional_pattern_mse - conditional_mapping_mse
                handle.write("mapping_conditional/pattern_only_mse: {:.10g}\n".format(conditional_pattern_mse))
                handle.write("mapping_conditional/mapping_mse: {:.10g}\n".format(conditional_mapping_mse))
                handle.write("mapping_conditional/gain: {:.10g}\n".format(conditional_gain))
                handle.write("mapping_conditional/gain_percent: {:.10g}\n".format(
                    100.0 * conditional_gain / max(conditional_pattern_mse, 1e-12)
                ))
                for stat_name, stat_value in (
                    ("mean", sample_mapping_gain.mean()), ("median", np.median(sample_mapping_gain)),
                    ("p10", np.quantile(sample_mapping_gain, 0.1)),
                    ("p50", np.quantile(sample_mapping_gain, 0.5)),
                    ("p90", np.quantile(sample_mapping_gain, 0.9)),
                    ("fraction_gain_positive", (sample_mapping_gain > 0).mean()),
                ):
                    handle.write("mapping_conditional/sample_{}: {:.10g}\n".format(stat_name, float(stat_value)))
                sample_mse = ((preds - trues) ** 2).mean(axis=(1, 2))
                sample_mae = np.abs(preds - trues).mean(axis=(1, 2))
                shuffled_mapping_mse = float(np.mean((shuffled_mapping_prediction - trues) ** 2))
                handle.write("mapping_shuffle/normal_mapping_mse: {:.10g}\n".format(float(mse)))
                handle.write("mapping_shuffle/shuffled_mapping_mse: {:.10g}\n".format(shuffled_mapping_mse))
                handle.write("mapping_shuffle/delta_mse: {:.10g}\n".format(shuffled_mapping_mse - float(mse)))
                edge_gain = module.pmg.mapping_graph.predictive_gain.detach().cpu().numpy()
                if edge_gain.size:
                    for stat_name, stat_value in (
                        ("mean", edge_gain.mean()), ("median", np.median(edge_gain)),
                        ("p10", np.quantile(edge_gain, 0.1)), ("p50", np.quantile(edge_gain, 0.5)),
                        ("p90", np.quantile(edge_gain, 0.9)),
                        ("fraction_positive", (edge_gain > 0).mean()),
                    ):
                        handle.write("edge_predictive_gain/{}: {:.10g}\n".format(stat_name, float(stat_value)))
                    edge_stability = module.pmg.mapping_graph.stability.detach().cpu().numpy()
                    edge_support = module.pmg.mapping_graph.window_support.detach().cpu().numpy()
                    handle.write("edge_predictive_gain/spearman_mapping_stability: {:.10g}\n".format(
                        _safe_correlation(edge_stability, edge_gain, spearman=True)
                    ))
                    handle.write("edge_predictive_gain/spearman_edge_support: {:.10g}\n".format(
                        _safe_correlation(edge_support, edge_gain, spearman=True)
                    ))
                for evidence_name, evidence in (
                    ("c_pat", c_pat_samples), ("c_map", c_map_samples), ("c_inv", c_inv_samples),
                    ("graph_gate", graph_gate_samples), ("correction_norm", correction_norm_samples),
                    ("mapping_deviation_norm", mapping_deviation_norm_samples),
                ):
                    handle.write("correlation/pearson_{}_sample_mse: {:.10g}\n".format(
                        evidence_name, _safe_correlation(evidence, sample_mse)
                    ))
                    handle.write("correlation/spearman_{}_sample_mse: {:.10g}\n".format(
                        evidence_name, _safe_correlation(evidence, sample_mse, spearman=True)
                    ))
                if is_raw:
                    handle.write("correlation/spearman_c_pat_raw_deviation_norm: {:.10g}\n".format(
                        _safe_correlation(c_pat_samples, raw_deviation_norm_samples, spearman=True)
                    ))
                    handle.write("correlation/spearman_raw_variation_zvar_norm: {:.10g}\n".format(
                        _safe_correlation(raw_deviation_norm_samples, zvar_norm_samples, spearman=True)
                    ))
                    handle.write("correlation/spearman_raw_variation_env_gain: {:.10g}\n".format(
                        _safe_correlation(raw_deviation_norm_samples, sample_env_gain, spearman=True)
                    ))
                    for component_name, component_values in (
                        ("shape_variation_norm", raw_component_samples["raw_shape_variation"]),
                        ("abs_d_shift", raw_component_samples["raw_abs_shift"]),
                        ("abs_d_scale", raw_component_samples["raw_abs_scale"]),
                        ("u_shift", raw_component_samples["raw_u_shift"]),
                        ("u_scale", raw_component_samples["raw_u_scale"]),
                        ("a_var", raw_component_samples["raw_a_var"]),
                    ):
                        handle.write("correlation/spearman_{}_sample_mse: {:.10g}\n".format(
                            component_name,
                            _safe_correlation(component_values, sample_mse, spearman=True),
                        ))
                    for component_name, component_values in (
                        ("shape_variation_norm", raw_component_samples["raw_shape_variation"]),
                        ("u_shift", raw_component_samples["raw_u_shift"]),
                        ("u_scale", raw_component_samples["raw_u_scale"]),
                        ("a_var", raw_component_samples["raw_a_var"]),
                    ):
                        handle.write("correlation/spearman_{}_env_gain: {:.10g}\n".format(
                            component_name,
                            _safe_correlation(component_values, sample_env_gain, spearman=True),
                        ))
                    for evidence_name, evidence_values, gain_name in (
                        ("shape_variation_norm", raw_component_samples["raw_shape_variation"], "shape"),
                        ("abs_d_scale", raw_component_samples["raw_abs_scale"], "scale"),
                        ("u_scale", raw_component_samples["raw_u_scale"], "scale"),
                        ("abs_d_shift", raw_component_samples["raw_abs_shift"], "shift"),
                        ("u_shift", raw_component_samples["raw_u_shift"], "shift"),
                    ):
                        handle.write("typed_correlation/spearman_{}_gain_{}: {:.10g}\n".format(
                            evidence_name, gain_name,
                            _safe_correlation(evidence_values, typed_gains[gain_name], spearman=True),
                        ))
                for confidence_name, confidence in (
                    ("c_shape", c_shape_samples), ("c_rec", c_rec_samples),
                    ("c_pred", c_pred_samples), ("c_pat", c_pat_samples),
                ):
                    handle.write("correlation/spearman_{}_sample_mse: {:.10g}\n".format(
                        confidence_name, _safe_correlation(confidence, sample_mse, spearman=True)
                    ))
                    handle.write("correlation/spearman_{}_sample_mae: {:.10g}\n".format(
                        confidence_name, _safe_correlation(confidence, sample_mae, spearman=True)
                    ))
                    for quantile_index, selected_indices in enumerate(
                        np.array_split(np.argsort(confidence, kind="stable"), 5), start=1
                    ):
                        prefix = "{}/quantile_Q{}".format(confidence_name, quantile_index)
                        handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                        handle.write("{}_MSE: {:.10g}\n".format(prefix, float(sample_mse[selected_indices].mean())))
                        handle.write("{}_MAE: {:.10g}\n".format(prefix, float(sample_mae[selected_indices].mean())))
                        handle.write("{}_mean: {:.10g}\n".format(prefix, float(confidence[selected_indices].mean())))
                for mapping_name, mapping_values in (
                    ("c_map", c_map_samples), ("u_map", mapping_novelty_samples),
                ):
                    for quantile_index, selected_indices in enumerate(
                        np.array_split(np.argsort(mapping_values, kind="stable"), 5), start=1
                    ):
                        prefix = "mapping_conditional/{}/Q{}".format(mapping_name, quantile_index)
                        handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                        handle.write("{}_pattern_only_mse: {:.10g}\n".format(
                            prefix, float(mapping_pattern_error[selected_indices].mean())
                        ))
                        handle.write("{}_mapping_mse: {:.10g}\n".format(
                            prefix, float(mapping_conditioned_error[selected_indices].mean())
                        ))
                        handle.write("{}_mapping_gain: {:.10g}\n".format(
                            prefix, float(sample_mapping_gain[selected_indices].mean())
                        ))
                        handle.write("{}_final_model_mse: {:.10g}\n".format(
                            prefix, float(sample_mse[selected_indices].mean())
                        ))
                for cpat_quantile, cpat_indices in enumerate(
                    np.array_split(np.argsort(c_pat_samples, kind="stable"), 5), start=1
                ):
                    local_order = cpat_indices[np.argsort(c_map_samples[cpat_indices], kind="stable")]
                    low_indices, high_indices = np.array_split(local_order, 2)
                    for level, selected_indices in (("low", low_indices), ("high", high_indices)):
                        prefix = "mapping_matched/cpat_Q{}/cmap_{}".format(cpat_quantile, level)
                        handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                        handle.write("{}_sample_mse: {:.10g}\n".format(
                            prefix, float(sample_mse[selected_indices].mean())
                        ))
                        handle.write("{}_mapping_gain: {:.10g}\n".format(
                            prefix, float(sample_mapping_gain[selected_indices].mean())
                        ))
                bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
                for lower, upper in bins:
                    selected = (c_inv_samples >= lower) & (c_inv_samples < upper)
                    label = "{:.1f}_{:.1f}".format(lower, min(upper, 1.0))
                    handle.write("cinv/bucket_{}_count: {}\n".format(label, int(selected.sum())))
                    handle.write("cinv/bucket_{}_MSE: {}\n".format(
                        label, "{:.10g}".format(float(sample_mse[selected].mean())) if selected.any() else "nan"
                    ))
                    handle.write("cinv/bucket_{}_MAE: {}\n".format(
                        label, "{:.10g}".format(float(sample_mae[selected].mean())) if selected.any() else "nan"
                    ))
                # Rank-based quintiles remain well-defined when c_pat occupies a narrow range.
                for quantile_index, selected_indices in enumerate(
                    np.array_split(np.argsort(c_pat_samples, kind="stable"), 5), start=1
                ):
                    prefix = "cpat/quantile_Q{}".format(quantile_index)
                    handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                    quantile_metrics = [
                        ("MSE", sample_mse), ("MAE", sample_mae),
                        ("c_pat_mean", c_pat_samples), ("c_map_mean", c_map_samples),
                        ("graph_gate_mean", graph_gate_samples),
                        ("correction_norm_mean", correction_norm_samples),
                    ]
                    if is_raw:
                        quantile_metrics.append(("raw_deviation_norm", raw_deviation_norm_samples))
                    quantile_metrics.extend([
                        ("inv_MSE", sample_inv_mse),
                        ("env_MSE", sample_env_mse),
                        ("env_gain", sample_env_gain),
                    ])
                    for metric_name, metric_values in quantile_metrics:
                        value = float(metric_values[selected_indices].mean()) if selected_indices.size else float("nan")
                        handle.write("{}_{}: {:.10g}\n".format(prefix, metric_name, value))
                if is_raw:
                    for evidence_name, evidence_values, operator_name, operator_mse, operator_gain in (
                        ("shape_variation_norm", raw_component_samples["raw_shape_variation"],
                         "shape", sample_shape_mse, typed_gains["shape"]),
                        ("u_scale", raw_component_samples["raw_u_scale"],
                         "scale", sample_scale_mse, typed_gains["scale"]),
                        ("u_shift", raw_component_samples["raw_u_shift"],
                         "shift", sample_shift_mse, typed_gains["shift"]),
                    ):
                        for quantile_index, selected_indices in enumerate(
                            np.array_split(np.argsort(evidence_values, kind="stable"), 5), start=1
                        ):
                            prefix = "typed_quantile/{}/Q{}".format(evidence_name, quantile_index)
                            handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                            handle.write("{}_inv_MSE: {:.10g}\n".format(
                                prefix, float(sample_inv_mse[selected_indices].mean())
                            ))
                            handle.write("{}_{}_MSE: {:.10g}\n".format(
                                prefix, operator_name, float(operator_mse[selected_indices].mean())
                            ))
                            handle.write("{}_{}_gain: {:.10g}\n".format(
                                prefix, operator_name, float(operator_gain[selected_indices].mean())
                            ))
                    for component_name, component_values in (
                        ("shape_variation", raw_component_samples["raw_shape_variation"]),
                        ("u_shift", raw_component_samples["raw_u_shift"]),
                        ("u_scale", raw_component_samples["raw_u_scale"]),
                        ("a_var", raw_component_samples["raw_a_var"]),
                    ):
                        for quantile_index, selected_indices in enumerate(
                            np.array_split(np.argsort(component_values, kind="stable"), 5), start=1
                        ):
                            prefix = "raw_variation/{}/Q{}".format(component_name, quantile_index)
                            handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                            for metric_name, metric_values in (
                                ("inv_MSE", sample_inv_mse), ("env_MSE", sample_env_mse),
                                ("env_gain", sample_env_gain), ("final_MSE", sample_mse),
                            ):
                                handle.write("{}_{}: {:.10g}\n".format(
                                    prefix, metric_name, float(metric_values[selected_indices].mean())
                                ))
                    for quantile_index, selected_indices in enumerate(
                        np.array_split(np.argsort(raw_deviation_norm_samples, kind="stable"), 5), start=1
                    ):
                        prefix = "raw_variation/quantile_Q{}".format(quantile_index)
                        handle.write("{}_count: {}\n".format(prefix, int(selected_indices.size)))
                        for metric_name, metric_values in (
                            ("inv_MSE", sample_inv_mse), ("env_MSE", sample_env_mse),
                            ("env_gain", sample_env_gain), ("final_MSE", sample_mse),
                            ("raw_variation_norm", raw_deviation_norm_samples),
                            ("c_pat_mean", c_pat_samples),
                        ):
                            handle.write("{}_{}: {:.10g}\n".format(
                                prefix, metric_name, float(metric_values[selected_indices].mean())
                            ))
                if is_a1:
                    handle.write("A1/Z_inv_only_MSE: {:.10g}\n".format(float(inv_mse)))
                    handle.write("A1/Z_inv_only_MAE: {:.10g}\n".format(float(inv_mae)))
                    if baseline_preds:
                        baseline_prediction = np.concatenate(baseline_preds)
                        baseline_mae, baseline_mse, _, _, _ = metric(baseline_prediction, trues)
                        handle.write("A1/original_A0_MSE: {:.10g}\n".format(float(baseline_mse)))
                        handle.write("A1/original_A0_MAE: {:.10g}\n".format(float(baseline_mae)))
                    sample_error = sample_mse
                    for lower, upper in bins:
                        selected = (c_inv_samples >= lower) & (c_inv_samples < upper)
                        label = "{:.1f}_{:.1f}".format(lower, min(upper, 1.0))
                        handle.write("A1/c_inv_bin_{}_count: {}\n".format(label, int(selected.sum())))
                        handle.write("A1/c_inv_bin_{}_forecast_MSE: {}\n".format(
                            label,
                            "{:.10g}".format(float(sample_error[selected].mean())) if selected.any() else "nan",
                        ))
                warmup_path = os.path.join(self.args.checkpoints, setting, "stage0_warmup.txt")
                if os.path.isfile(warmup_path):
                    handle.write("\n[stage0_warmup]\n")
                    with open(warmup_path, "r", encoding="utf-8") as warmup_handle:
                        handle.write(warmup_handle.read())
            print("Saved FPem diagnostics: {}".format(diagnostics_path))
            if is_raw:
                def _read_text_metrics(path):
                    values = {}
                    if os.path.isfile(path):
                        with open(path, "r", encoding="utf-8") as source:
                            for line in source:
                                if ":" not in line:
                                    continue
                                key, value = line.strip().split(":", 1)
                                values[key] = value.strip()
                    return values

                embedding_candidates = [
                    candidate for candidate in glob.glob(
                        "./results/*pmg_emb_A3*zinv_C2_stablerelcorr*"
                    ) if "smoke" not in candidate
                ]
                a0_candidates = glob.glob("./results/*A0_PatchTST*fpem_pmg_A0*")
                comparison_path = os.path.join(folder_path, "embedding_vs_raw_report.txt")
                with open(comparison_path, "w", encoding="utf-8") as report:
                    report.write("A0 PatchTST:\n")
                    if a0_candidates:
                        a0_folder = max(a0_candidates, key=os.path.getmtime)
                        a0_metrics = np.load(os.path.join(a0_folder, "metrics.npy"))
                        report.write("MSE: {:.10g}\nMAE: {:.10g}\n\n".format(float(a0_metrics[1]), float(a0_metrics[0])))
                    else:
                        report.write("MSE: 0.390853\nMAE: unavailable\n\n")
                    report.write("R0 embedding C2:\n")
                    if embedding_candidates:
                        emb_folder = max(embedding_candidates, key=os.path.getmtime)
                        emb_metrics = np.load(os.path.join(emb_folder, "metrics.npy"))
                        emb_diag = _read_text_metrics(os.path.join(emb_folder, "fpem_diagnostics.txt"))
                        for key, value in (
                            ("MSE", "{:.10g}".format(float(emb_metrics[1]))),
                            ("MAE", "{:.10g}".format(float(emb_metrics[0]))),
                            ("num_patterns", emb_diag.get("graph/num_patterns", emb_diag.get("pattern/num_active", "nan"))),
                            ("num_edges", emb_diag.get("graph/num_mappings", emb_diag.get("mapping/num_active_edges", "nan"))),
                            ("c_pat_mean", emb_diag.get("pattern/c_mean", "nan")),
                            ("c_pat_std", emb_diag.get("pattern/c_std", emb_diag.get("cinv/std", "nan"))),
                            ("c_map_mean", emb_diag.get("mapping/c_mean", "nan")),
                            ("c_map_std", emb_diag.get("mapping/c_std", "nan")),
                            ("spearman_c_pat_error", emb_diag.get("correlation/spearman_c_pat_sample_mse", "nan")),
                        ):
                            report.write("{}: {}\n".format(key, value))
                    report.write("\nR1 raw C2:\n")
                    for key, value in (
                        ("MSE", mse), ("MAE", mae),
                        ("num_patterns", module.pmg.pattern_graph.num_active),
                        ("num_edges", module.pmg.mapping_graph.num_active),
                        ("c_pat_mean", c_pat_samples.mean()), ("c_pat_std", c_pat_samples.std()),
                        ("c_map_mean", c_map_samples.mean()), ("c_map_std", c_map_samples.std()),
                        ("spearman_c_pat_error", _safe_correlation(c_pat_samples, sample_mse, spearman=True)),
                    ):
                        report.write("{}: {:.10g}\n".format(key, float(value)))
                print("Saved embedding/raw comparison: {}".format(comparison_path))

        return
