#!/usr/bin/env python3
import argparse,csv
from pathlib import Path

DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','ExchangeRate','Weather','Electricity','Traffic')
PREDS=(96,336,720)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--backbones',default='patchtst'); p.add_argument('--output',required=True); a=p.parse_args()
    root=Path(a.root); rows=[]
    for backbone in a.backbones.split(','):
        for dataset in DATASETS:
            for pred in PREDS:
                case=root/backbone/dataset/f'pred_{pred}'
                if not (case/'best_config.json').is_file(): rows.append((backbone,dataset,pred))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['# backbone','dataset','pred_len']); w.writerows(rows)
    print(f'missing={len(rows)} output={a.output}')
if __name__=='__main__': main()
