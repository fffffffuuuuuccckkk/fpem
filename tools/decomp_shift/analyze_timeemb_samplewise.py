import argparse, copy, json, os, sys, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT='/data/OuXiaoyu/TimeEmb-main/TimeEmb-main'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from exp.exp_main import Exp_Main


def setting(a):
    return '{}_{}_{}_ft{}_sl{}_pl{}_hi{}_di{}_hl{}_dl{}_dm{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(a.model_id,a.model,a.data,a.features,a.seq_len,a.pred_len,a.use_hour_index,a.use_day_index,a.hour_length,a.day_length,a.d_model,a.auxi_lambda,a.rec_lambda,a.auxi_loss,a.module_first,a.des,0)

def eval_ckpt(args, ckpt, force_hour=None):
    aa=copy.copy(args)
    if force_hour is not None: aa.use_hour_index=force_hour
    exp=Exp_Main(aa); exp.model.load_state_dict(torch.load(ckpt,map_location='cpu')); exp.model.to(exp.device).eval()
    data,loader=exp._get_data('test'); losses=[]
    with torch.no_grad():
        for bx,by,bxm,bym,hi,di in loader:
            bx=bx.float().to(exp.device); by=by.float().to(exp.device); hi=hi.int().to(exp.device)
            if torch.is_tensor(di) and torch.all(di!=-1): di=di.int().to(exp.device)
            out=exp.model(bx,hi,di)[:,-aa.pred_len:,:]
            true=by[:,-aa.pred_len:,:]
            losses.append((out-true).abs().mean((1,2)).cpu().numpy())
    return np.concatenate(losses)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--full_ckpt',required=True); p.add_argument('--nohour_ckpt',default=''); p.add_argument('--out_dir',required=True); p.add_argument('--model_id',default='ETTh1_96_48_timeemb_full_0823'); p.add_argument('--des',default='timeemb_full_0823'); p.add_argument('--root_path',default='./dataset/'); p.add_argument('--data_path',default='ETTh1.csv'); p.add_argument('--data',default='ETTh1'); p.add_argument('--features',default='M'); p.add_argument('--target',default='OT'); p.add_argument('--freq',default='h'); p.add_argument('--checkpoints',default='./checkpoints/'); p.add_argument('--seq_len',type=int,default=96); p.add_argument('--label_len',type=int,default=0); p.add_argument('--pred_len',type=int,default=48); p.add_argument('--use_revin',type=int,default=1); p.add_argument('--use_hour_index',type=int,default=1); p.add_argument('--use_day_index',type=int,default=0); p.add_argument('--hour_length',type=int,default=24); p.add_argument('--day_length',type=int,default=7); p.add_argument('--rec_lambda',type=float,default=0.0); p.add_argument('--auxi_lambda',type=float,default=1.0); p.add_argument('--auxi_loss',default='MAE'); p.add_argument('--module_first',type=int,default=1); p.add_argument('--model',default='TimeEmb'); p.add_argument('--embed',default='timeF'); p.add_argument('--seasonal_patterns',default='Monthly'); p.add_argument('--enc_in',type=int,default=7); p.add_argument('--dec_in',type=int,default=7); p.add_argument('--c_out',type=int,default=7); p.add_argument('--d_model',type=int,default=512); p.add_argument('--gpu',type=int,default=0); p.add_argument('--use_gpu',type=bool,default=True); p.add_argument('--use_multi_gpu',action='store_true',default=False); p.add_argument('--devices',default='0'); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--num_workers',type=int,default=0); p.add_argument('--is_training',type=int,default=0); p.add_argument('--itr',type=int,default=1); p.add_argument('--train_epochs',type=int,default=1); p.add_argument('--patience',type=int,default=3); p.add_argument('--learning_rate',type=float,default=1e-4); p.add_argument('--lradj',default='type3'); p.add_argument('--pct_start',type=float,default=0.3); p.add_argument('--use_amp',action='store_true',default=False); p.add_argument('--test_flop',action='store_true',default=False); p.add_argument('--output_attention',action='store_true',default=False); p.add_argument('--add_noise',action='store_true',default=False); p.add_argument('--noise_amp',type=float,default=0); p.add_argument('--noise_freq_percentage',type=float,default=0.05); p.add_argument('--auxi_mode',default='fft'); p.add_argument('--auxi_type',default='complex'); p.add_argument('--leg_degree',type=int,default=2)
    a=p.parse_args(); a.device=torch.device('cuda:%d'%a.gpu if torch.cuda.is_available() and a.use_gpu else 'cpu')
    os.makedirs(a.out_dir,exist_ok=True)
    full=eval_ckpt(a,a.full_ckpt,1); nohour_infer=eval_ckpt(a,a.full_ckpt,0)
    rows=[]; summary={'n':int(len(full)),'full_hour_mae':float(full.mean()),'same_ckpt_nohour_mae':float(nohour_infer.mean()),'hour_better_ratio':float((full<nohour_infer).mean()),'nohour_infer_better_ratio':float((nohour_infer<=full).mean())}
    cols=['sample','mae_full_hour','mae_same_ckpt_nohour','better_hour_or_nohour']
    trained_nohour=None
    if a.nohour_ckpt:
        trained_nohour=eval_ckpt(a,a.nohour_ckpt,0); summary.update({'trained_nohour_mae':float(trained_nohour.mean()),'trained_nohour_better_than_full_ratio':float((trained_nohour<full).mean()),'full_better_than_trained_nohour_ratio':float((full<=trained_nohour).mean())}); cols += ['mae_trained_nohour','better_full_or_trained_nohour']
    with open(os.path.join(a.out_dir,'samplewise_timeemb_compare.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(cols)
        for i in range(len(full)):
            r=[i,float(full[i]),float(nohour_infer[i]),'hour' if full[i]<nohour_infer[i] else 'nohour_infer']
            if trained_nohour is not None: r += [float(trained_nohour[i]),'trained_nohour' if trained_nohour[i]<full[i] else 'full']
            w.writerow(r)
    with open(os.path.join(a.out_dir,'summary.json'),'w') as f: json.dump(summary,f,indent=2)
    labels=['hour','same_ckpt_nohour']; vals=[summary['hour_better_ratio'],summary['nohour_infer_better_ratio']]
    if trained_nohour is not None: labels.append('trained_nohour_vs_full'); vals.append(summary['trained_nohour_better_than_full_ratio'])
    plt.figure(figsize=(6,4)); plt.bar(labels,vals); plt.ylim(0,1); plt.tight_layout(); plt.savefig(os.path.join(a.out_dir,'samplewise_ratio.png'),dpi=160)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()