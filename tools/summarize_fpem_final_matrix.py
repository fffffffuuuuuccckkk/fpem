#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

FIELDS = ['backbone','dataset','pred_len','K','environment_mode','decomposer_lr',
          'environment_classifier_lr','future_zvar_lr','gamma_beta_lr','reliability_lr',
          'A0_MSE','Zinv_MSE','raw_MSE','gated_MSE','best_final_MSE',
          'best_prediction_source','selection_source','metrics_path']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--csv',required=True); p.add_argument('--txt',required=True); a=p.parse_args()
    rows=[]
    for path in sorted(Path(a.root).rglob('best_config.json')):
        row=json.loads(path.read_text()); rows.append({k:row.get(k,'') for k in FIELDS})
    Path(a.csv).parent.mkdir(parents=True,exist_ok=True)
    with open(a.csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    lines=['\t'.join(FIELDS)]+['\t'.join(str(r[k]) for k in FIELDS) for r in rows]
    Path(a.txt).write_text('\n'.join(lines)+'\n')
    print(f'wrote {len(rows)} completed cases')
if __name__=='__main__': main()
