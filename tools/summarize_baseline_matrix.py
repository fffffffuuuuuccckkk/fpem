#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--csv',required=True); p.add_argument('--txt',required=True); a=p.parse_args(); rows=[]
    for f in sorted(Path(a.root).rglob('metrics.json')):
        d=json.loads(f.read_text()); rel=f.relative_to(a.root).parts
        if len(rel)>=4: rows.append({'method':rel[0],'dataset':rel[1],'pred_len':rel[2].replace('pred_',''),'MSE':d['MSE'],'MAE':d['MAE']})
    fields=['method','dataset','pred_len','MSE','MAE']; Path(a.csv).parent.mkdir(parents=True,exist_ok=True)
    with open(a.csv,'w',newline='') as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    Path(a.txt).write_text('\n'.join(['\t'.join(fields)]+['\t'.join(str(r[x]) for x in fields) for r in rows])+'\n')
if __name__=='__main__': main()
