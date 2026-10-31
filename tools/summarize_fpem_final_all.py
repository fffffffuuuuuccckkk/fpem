#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
FIELDS=['Method','Backbone','Dataset','Pred','K','A0','Zinv','Raw','Gated','Best','best_prediction_source','selection_source','source']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--roots',nargs='+',required=True); p.add_argument('--output-root',required=True); a=p.parse_args(); rows=[]
    for root_name in a.roots:
        root=Path(root_name)
        for f in root.rglob('best_config.json'):
            d=json.loads(f.read_text()); rows.append({'Method':'FPEM','Backbone':d.get('backbone'),'Dataset':d.get('dataset'),'Pred':d.get('pred_len'),'K':d.get('K'),'A0':d.get('A0_MSE'),'Zinv':d.get('Zinv_MSE'),'Raw':d.get('raw_MSE'),'Gated':d.get('gated_MSE'),'Best':d.get('best_final_MSE'),'best_prediction_source':d.get('best_prediction_source'),'selection_source':d.get('selection_source'),'source':str(f)})
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); rows.sort(key=lambda r:(str(r['Backbone']),str(r['Dataset']),int(r['Pred'])))
    with open(out/'final_matrix.csv','w',newline='') as h: w=csv.DictWriter(h,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    (out/'final_matrix.txt').write_text('\n'.join(['\t'.join(FIELDS)]+['\t'.join(str(r.get(x,'')) for x in FIELDS) for r in rows])+'\n')
    with open(out/'fpem_best_configs.csv','w',newline='') as h: w=csv.DictWriter(h,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f'completed FPEM cases={len(rows)}')
if __name__=='__main__': main()
