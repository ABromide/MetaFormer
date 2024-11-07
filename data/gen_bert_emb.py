import os
import pickle
import torch
from transformers import *
from tqdm import tqdm
root="datasets/cub-200"
tokenizer=AutoTokenizer.from_pretrained("../bert-base-uncased/")
model=BertModel.from_pretrained("../bert-base-uncased/").to("cuda:0")
text_root="datasets/cub-200/text_c10/"
embedding_root="datasets/cub-200/bert_embedding_cub"
model.eval()
with torch.no_grad():
    with open(os.path.join(os.path.join(root,'CUB_200_2011'),'images.txt'),'r') as f:
        lines=f.readlines()
        for line in tqdm(lines):
            image_id,file_name = line.split()
            text_file = file_name.replace('.jpg','.txt')
            text_file = text_root + text_file
            text_list = []
            with open(text_file,'r') as f_text:
                for line in f_text:
                    line = line.encode(encoding='UTF-8',errors='strict')
                    line = line.replace(b'\xef\xbf\xbd\xef\xbf\xbd',b' ')
                    line = line.decode('UTF-8','strict')
                    text_list.append(line) 
            inputs = tokenizer(text_list, return_tensors="pt",padding="max_length",truncation=True, max_length=32)
            for k in inputs:
                if isinstance(inputs[k],torch.Tensor):
                    inputs[k]=inputs[k].to("cuda:0")
            outputs = model(**inputs)
            embedding_mean = outputs[1].mean(dim=0).reshape(1,-1).detach().cpu().numpy()
            embedding_full = outputs[1].detach().cpu().numpy()
            embedding_words = outputs[0].detach().cpu().numpy()
            data_dict = {
                'embedding_mean':embedding_mean,
                'embedding_full':embedding_full,
                'embedding_words':embedding_words,
            }
            class_name,image_name = file_name.split('/')
            if not os.path.exists(os.path.join(embedding_root,class_name)):
                os.makedirs(os.path.join(embedding_root,class_name))
            embedding_file_path = os.path.join(os.path.join(embedding_root,class_name),image_name.replace('.jpg','.pickle'))
            with open(embedding_file_path,'wb') as f_write:
                pickle.dump(data_dict,f_write)