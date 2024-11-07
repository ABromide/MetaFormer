import torch
from .MetaFG_meta import MetaFG_Meta
import copy
import torch.nn.functional as F
from typing import Callable
class EnsembleModel(torch.nn.Module):
    def __init__(self,meta_fg_model_builder:Callable[[],MetaFG_Meta]) -> None:
        super().__init__()
        self.meta_fg_model_A=meta_fg_model_builder()
        self.meta_fg_model_B=meta_fg_model_builder()
        self.meta_fg_model_C=meta_fg_model_builder()
        self.embeddingA=torch.nn.Embedding(300,self.meta_fg_model_A.meta_dims[0])
        # self.embeddingB=torch.nn.Embedding(300,64)
        # self.embeddingC=torch.nn.Embedding(300,64)
    def forward(self,picA:torch.Tensor,picB:torch.Tensor,picC:torch.Tensor,meta:torch.Tensor,weight:torch.Tensor):
        meta_emb1=self.embeddingA(meta)
        # meta_emb2=self.embeddingA(meta)
        # meta_emb3=self.embeddingA(meta)
        x1=self.meta_fg_model_A(picA,meta_emb1)
        x2=self.meta_fg_model_B(picB,meta_emb1)
        x3=self.meta_fg_model_C(picC,meta_emb1)
        x1=F.softmax(x1,1)
        x2=F.softmax(x2,1)
        x3=F.softmax(x3,1)
        res=x1*(weight[:,0].unsqueeze(1))+x2*(weight[:,1].unsqueeze(1))+x3*(weight[:,2].unsqueeze(1))
        # res=x1+x2+x3
        return res
