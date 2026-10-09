#!/usr/bin/env python3
import argparse,csv,re
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); root=Path(a.root); rows=[]
    for dataset in ('ETTh1','ETTh2','ETTm1','ETTm2','ExchangeRate','Weather','Electricity','Traffic'):
        for pred in (96,336,720):
            d=root/dataset/f'pred_{pred}'; status='unavailable'; mse=mae=''
            log=d/'stage1.log'
            if log.is_file():
                matches=re.findall(r'mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)',log.read_text(errors='ignore'))
                if matches: mse,mae=matches[-1]; status='complete'
            rows.append({'dataset':dataset,'pred_len':pred,'status':status,'MSE':mse,'MAE':mae})
    fields=['dataset','pred_len','status','MSE','MAE']
    with open(root/'foil_summary.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (root/'foil_summary.txt').write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(r[x]) for x in fields) for r in rows])+'\n')
if __name__=='__main__': main()
