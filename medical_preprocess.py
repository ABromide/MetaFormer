
from dataclasses import dataclass


path="datasets/medical/meta.csv"

@dataclass
class InputFeature:
    pid:int=1
    diagnosis:str=""
    id_:str=""
    name:str=""
    gender:str=""
    age:int=1
    t_spot:int=1
    tb_ab_igg:int=1
    tb_ab_igm:int=1
    pdd:int=1
    abdominal_pain:int=1
    diarrhea:int=1
    rectum:int=1
    terminal_ileum:int=1
    ileocecus:int=1
    other:int=1

