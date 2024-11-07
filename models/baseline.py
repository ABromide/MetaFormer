import torch
from .MetaFG_meta import MetaFG_Meta
class BaselineModel(torch.nn.Module):
    def __init__(self,meta_fg_model:MetaFG_Meta) -> None:
        super().__init__()
        self.meta_fg_model=meta_fg_model
        self.embedding=torch.nn.Embedding(130,768)
    def forward(self,pic1:torch.Tensor,pic2:torch.Tensor,meta:torch.Tensor):
        meta_emb=self.embedding(meta)
        x1=self.meta_fg_model(pic1,meta_emb)
        x2=self.meta_fg_model(pic2,meta_emb)
        res=x1+x2
        return res