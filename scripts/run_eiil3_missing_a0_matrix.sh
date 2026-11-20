#!/usr/bin/env bash
# Build audited, absent A0 references once; never overwrite a valid checkpoint.
set -euo pipefail
gpu="$1"
plan="$2"
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
asset_root="$root/results/fpem_eiil3_mse_priority_20261112/a0_assets"
source "$root/scripts/fpem_final/common.sh"
mkdir -p "$asset_root/queue_logs"
cd "$root"
while IFS=, read -r backbone dataset horizon; do
  [[ -z "$backbone" || "$backbone" == \#* ]] && continue
  read -r data_root data_class data_path freq batch cycle channels \
    <<<"$(dataset_config "$dataset")"
  case "$backbone" in
    patchtst) d_model=128; d_ff=256; heads=8; layers=1; dropout=.1 ;;
    itransformer) d_model=512; d_ff=512; heads=8; layers=3; dropout=.1 ;;
    moderntcn)
      d_model=64; heads=2; layers=1
      read -r patch_size patch_stride stages ffn_ratio dropout head_dropout \
        <<<"$(modern_tcn_config "$dataset")"
      d_ff=$((d_model * ffn_ratio)) ;;
    *) echo "unsupported backbone $backbone" >&2; exit 2 ;;
  esac
  output="$asset_root/$backbone/$dataset/pred_$horizon/seed_2021"
  checkpoint="$output/shared_reference.pt"
  log="$asset_root/queue_logs/${backbone}_${dataset}_${horizon}.log"
  mkdir -p "$output"
  if [[ -s "$checkpoint" && -s "$output/run_config.json" ]]; then
    echo "REUSED $backbone/$dataset/$horizon" >> "$log"
    continue
  fi
  # Small batches avoid memory spikes on 8 GiB GPUs; this is recorded in run_config.
  [[ "$data_class" == custom ]] && batch=1
  for attempt in 1 2; do
    echo "START $backbone/$dataset/$horizon attempt=$attempt" >> "$log"
    if CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/run_predictive_env_iv_patchtst.py \
        --backbone "$backbone" --dataset_name "$dataset" \
        --data_class "$data_class" --root_path "$data_root" \
        --data_path "$data_path" --freq "$freq" \
        --cyclenet_cycle_len "$cycle" --seq_len 96 --pred_len "$horizon" \
        --d_model "$d_model" --d_ff "$d_ff" --n_heads "$heads" \
        --e_layers "$layers" --dropout "$dropout" \
        --modern_tcn_patch_size "${patch_size:-8}" \
        --modern_tcn_patch_stride "${patch_stride:-4}" \
        --modern_tcn_num_stages "${stages:-1}" \
        --modern_tcn_ffn_ratio "${ffn_ratio:-1}" \
        --modern_tcn_head_dropout "${head_dropout:-0}" \
        --env_num 3 --seed 2021 --gpu 0 --batch_size "$batch" --num_workers 0 \
        --warmup_epochs 10 --lr 1e-4 --reference_checkpoint "$checkpoint" \
        --prepare_reference_only --output "$output" >> "$log" 2>&1; then
      if [[ -s "$checkpoint" && -s "$output/run_config.json" ]]; then
        sha256sum "$checkpoint" > "$output/A0.sha256"
        "$python" tools/build_eiil3_priority_case_manifest.py \
          --source-manifest results/fpem_eiil3_20261110/manifest.jsonl \
          --case-key "$backbone/$dataset/pred_$horizon/seed_2021" \
          --a0-checkpoint "$checkpoint" --template-config "$output/run_config.json" \
          --output-dir "$output" >> "$log" 2>&1
        echo "COMPLETE $backbone/$dataset/$horizon" >> "$log"
        break
      fi
    fi
    echo "FAILED $backbone/$dataset/$horizon attempt=$attempt" >> "$log"
    [[ "$attempt" -eq 1 ]] && sleep 60
  done
done < "$plan"
