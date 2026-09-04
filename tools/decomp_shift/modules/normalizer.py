import torch
from torch import nn
class RevIN(nn.Module):
    def __init__(self,num_features,axis=(1,2),eps=1e-5):
        super().__init__(); self.axis=axis; self.eps=eps; self.mean=None; self.std=None
    def forward(self,x,mode="norm"):
        if mode=="norm":
            self.mean=x.mean(dim=self.axis,keepdim=True).detach(); self.std=(x.var(dim=self.axis,keepdim=True,unbiased=False)+self.eps).sqrt().detach(); return (x-self.mean)/self.std
        return x*self.std+self.mean
    def normalize(self,x): return (x-self.mean)/self.std
