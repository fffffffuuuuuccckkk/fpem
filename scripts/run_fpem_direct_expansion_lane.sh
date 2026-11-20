#!/usr/bin/env bash
set -u -o pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2

lane="${1:?lane}"
gpu="${2:?gpu}"
python_bin="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root=results/fpem_performance_optimization_20261103/direct_expansion

case "${lane}" in
  it_a) backbone=itransformer; method=F03; cases=('ETTh1 336' 'ETTh1 720' 'ETTh2 336' 'ETTh2 720') ;;
  it_b) backbone=itransformer; method=F03; cases=('ETTm1 192' 'ETTm1 336' 'ETTm2 336' 'ETTm2 720') ;;
  it_c) backbone=itransformer; method=F03; cases=('ExchangeRate 96' 'ExchangeRate 192' 'ExchangeRate 336' 'ExchangeRate 720') ;;
  mt_a) backbone=moderntcn; method=P02; cases=('ETTm1 720' 'ETTm2 96' 'ETTm2 192' 'ETTm2 336') ;;
  mt_b) backbone=moderntcn; method=P02; cases=('ETTh1 336' 'ETTh1 720' 'ETTh2 336' 'ETTh2 720') ;;
  mt_c) backbone=moderntcn; method=P02; cases=('ExchangeRate 96' 'ExchangeRate 192' 'ExchangeRate 336' 'ExchangeRate 720') ;;
  pt_a) backbone=patchtst; method=P02; cases=('ETTh1 96' 'ETTh1 192' 'ETTh1 336' 'ETTh1 720') ;;
  pt_b) backbone=patchtst; method=P02; cases=('ETTh2 96' 'ETTh2 192' 'ETTh2 336' 'ETTh2 720') ;;
  pt_c) backbone=patchtst; method=P02; cases=('ETTm1 96' 'ETTm1 192' 'ETTm1 336' 'ETTm1 720') ;;
  pt_d) backbone=patchtst; method=P02; cases=('ETTm2 96' 'ETTm2 192' 'ETTm2 336' 'ETTm2 720') ;;
  pt_e) backbone=patchtst; method=P02; cases=('ExchangeRate 96' 'ExchangeRate 192' 'ExchangeRate 336' 'ExchangeRate 720') ;;
  *) echo "Unknown approved lane: ${lane}" >&2; exit 2 ;;
esac

lane_dir="${root}/lanes/${lane}"
mkdir -p "${lane_dir}"
if [[ -f "${lane_dir}/queue_complete" ]]; then
  exit 0
fi

for item in "${cases[@]}"; do
  read -r dataset horizon <<< "${item}"
  case_dir="${root}/${backbone}/${dataset}/pred_${horizon}"
  base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
  mkdir -p "${case_dir}"
  if [[ ! -f "${base}" ]]; then
    echo "Missing reusable A0 config: ${base}" >> "${case_dir}/queue.log"
    exit 2
  fi
  if [[ -f "${case_dir}/queue_complete" ]]; then
    continue
  fi
  echo "START ${backbone}/${dataset}/${horizon} on gpu=${gpu}" >> "${lane_dir}/queue.log"
  success=0
  for attempt in 1 2 3; do
    echo "attempt=${attempt} lane=${lane} gpu=${gpu} case=${backbone}/${dataset}/${horizon} method=${method}" >> "${case_dir}/queue.log"
    if PYTHONPATH=. "${python_bin}" -u tools/run_fpem_direct_invariant_search.py \
        --base_run_config "${base}" --output "${case_dir}" --gpu "${gpu}" \
        --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
        --methods "F00,${method}" --allow_legacy_reference_signature \
        >> "${case_dir}/queue.log" 2>&1; then
      if [[ -f "${case_dir}/F00/complete" && -f "${case_dir}/${method}/complete" ]]; then
        touch "${case_dir}/queue_complete"
        echo "COMPLETE ${backbone}/${dataset}/${horizon}" >> "${lane_dir}/queue.log"
        success=1
        break
      fi
    fi
    echo "attempt=${attempt} failed; retry after 60s" >> "${case_dir}/queue.log"
    sleep 60
  done
  if [[ "${success}" -ne 1 ]]; then
    echo "case failed after three attempts" >> "${case_dir}/queue.log"
    echo "FAILED ${backbone}/${dataset}/${horizon}" >> "${lane_dir}/queue.log"
    exit 1
  fi
done

touch "${lane_dir}/queue_complete"
