import argparse, json, os, sys, math, shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC='/data/OuXiaoyu'
sys.path.insert(0, SRC)
# provide minimal modules.normalizer.RevIN expected by knfmodels.py
if not os.path.exists('/data/OuXiaoyu/Time-Series-Library-FPEM/tools/decomp_shift/modules'):
    os.makedirs('/data/OuXiaoyu/Time-Series-Library-FPEM/tools/decomp_shift/modules', exist_ok=True)
open('/data/OuXiaoyu/Time-Series-Library-FPEM/tools/decomp_shift/modules/__init__.py','a').close()
with open('/data/OuXiaoyu/Time-Series-Library-FPEM/tools/decomp_shift/modules/normalizer.py','w') as f:
    f.write('''import torch\nfrom torch import nn\nclass RevIN(nn.Module):\n    def __init__(self,num_features,axis=(1,2),eps=1e-5):\n        super().__init__(); self.axis=axis; self.eps=eps; self.mean=None; self.std=None\n    def forward(self,x,mode=\"norm\"):\n        if mode==\"norm\":\n            self.mean=x.mean(dim=self.axis,keepdim=True).detach(); self.std=(x.var(dim=self.axis,keepdim=True,unbiased=False)+self.eps).sqrt().detach(); return (x-self.mean)/self.std\n        return x*self.std+self.mean\n    def normalize(self,x): return (x-self.mean)/self.std\n''')
sys.path.insert(0, '/data/OuXiaoyu/Time-Series-Library-FPEM/tools/decomp_shift')
from knfmodels import Koopman

class ETT(Dataset):
    def __init__(self, root, flag, seq_len=96, pred_len=48, features='M'):
        df=pd.read_csv(os.path.join(root,'ETTh1.csv'))
        vals=df[df.columns[1:]].values.astype('float32') if features=='M' else df[['OT']].values.astype('float32')
        b1=[0,12*30*24-seq_len,12*30*24+4*30*24-seq_len]; b2=[12*30*24,12*30*24+4*30*24,12*30*24+8*30*24]
        idx={'train':0,'val':1,'test':2}[flag]
        sc=StandardScaler().fit(vals[b1[0]:b2[0]]); vals=sc.transform(vals).astype('float32')
        self.x=vals[b1[idx]:b2[idx]]; self.seq_len=seq_len; self.pred_len=pred_len
    def __len__(self): return len(self.x)-self.seq_len-self.pred_len+1
    def __getitem__(self,i): return self.x[i:i+self.seq_len], self.x[i+self.seq_len:i+self.seq_len+self.pred_len]

def make_model(args, add_control=True):
    return Koopman(input_dim=1,input_length=args.seq_len,output_dim=1,num_steps=args.pred_len,
        encoder_hidden_dim=args.hidden,decoder_hidden_dim=args.hidden,encoder_num_layers=2,decoder_num_layers=2,
        latent_dim=args.latent,num_feats=args.enc_in,add_global_operator=True,add_control=add_control,
        control_num_layers=2,control_hidden_dim=args.hidden,use_revin=True,use_instancenorm=False,
        regularize_rank=False,num_sins=8,num_heads=1,transformer_dim=args.hidden,transformer_num_layers=1,dropout_rate=0.05)

def eval_model(model, loader, dev, disable_control=False):
    old=model.add_control; model.add_control = (old and not disable_control); model.eval(); losses=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.float().to(dev); y=y.float().to(dev); pred=model(x,y)[0][:,:y.shape[1]]
            losses.append((pred-y).abs().mean((1,2)).cpu().numpy())
    model.add_control=old
    return np.concatenate(losses)

def train(args, add_control, out):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=DataLoader(ETT(args.root,'train',args.seq_len,args.pred_len),batch_size=args.batch,shuffle=True,num_workers=0)
    va=DataLoader(ETT(args.root,'val',args.seq_len,args.pred_len),batch_size=args.batch,shuffle=False,num_workers=0)
    te=DataLoader(ETT(args.root,'test',args.seq_len,args.pred_len),batch_size=args.batch,shuffle=False,num_workers=0)
    m=make_model(args,add_control).to(dev); opt=torch.optim.Adam(m.parameters(),lr=args.lr); mse=nn.MSELoss(); best=1e9; bad=0
    os.makedirs(out,exist_ok=True)
    for ep in range(1,args.epochs+1):
        m.train(); losses=[]
        for x,y in tr:
            x=x.float().to(dev); y=y.float().to(dev); opt.zero_grad(); pred=m(x,y)[0][:,:y.shape[1]]; loss=mse(pred,y); loss.backward(); opt.step(); losses.append(loss.item())
        v=eval_model(m,va,dev,False).mean(); print('epoch',ep,'train',float(np.mean(losses)),'val_mae',float(v),flush=True)
        if v<best: best=v; bad=0; torch.save(m.state_dict(),os.path.join(out,'checkpoint.pth'))
        else: bad+=1
        if bad>=args.patience: break
    m.load_state_dict(torch.load(os.path.join(out,'checkpoint.pth'),map_location=dev)); return m,te,dev

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='./dataset/ETT-small'); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=32); p.add_argument('--seq_len',type=int,default=96); p.add_argument('--pred_len',type=int,default=48); p.add_argument('--enc_in',type=int,default=7); p.add_argument('--hidden',type=int,default=64); p.add_argument('--latent',type=int,default=32); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--patience',type=int,default=3); p.add_argument('--mode',choices=['full','nocontrol'],default='full')
    a=p.parse_args(); full=(a.mode=='full')
    m,te,dev=train(a,full,a.out); full_loss=eval_model(m,te,dev,False); off_loss=eval_model(m,te,dev,True)
    summary={'mode':a.mode,'n':int(len(full_loss)),'decomp_or_control_mae':float(full_loss.mean()),'no_control_infer_mae':float(off_loss.mean()),'decomp_better_ratio':float((full_loss<off_loss).mean()),'no_control_better_ratio':float((off_loss<=full_loss).mean())}
    import csv
    with open(os.path.join(a.out,'samplewise_control_compare.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['sample','mae_with_control','mae_without_control','better']); [w.writerow([i,float(full_loss[i]),float(off_loss[i]),'control' if full_loss[i]<off_loss[i] else 'no_control']) for i in range(len(full_loss))]
    with open(os.path.join(a.out,'summary.json'),'w') as f: json.dump(summary,f,indent=2)
    plt.figure(figsize=(5,4)); plt.bar(['with_control','no_control'],[summary['decomp_better_ratio'],summary['no_control_better_ratio']]); plt.ylim(0,1); plt.tight_layout(); plt.savefig(os.path.join(a.out,'samplewise_ratio.png'),dpi=160)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()