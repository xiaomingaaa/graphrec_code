#### 可复用的模块


import dgl
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
#triple_datapath='dataset/KG/triple.tsv'
#entities_datapath='dataset/KG/entities.tsv'
#### user item, 
#### 1. 从triple_set.tsv中取边
#### 2. 使用dgl构建成graph对象
def construst_kg():
    ### entity_name, entity_id
    g = dgl.graph()
    triple_set=pd.read_csv('dataset/KG/triple.tsv')
    entities_dict=pd.read_csv('dataset/KG/entities.tsv')
    #构建dgl图
    g.add_nodes(len(entities_dict))
    g=dgl.ndata['feat']=torch.LongTensor(entities_dict[:,1])
    #添加边到dgl图
    g.add_edges(triple_set['user_id'],triple_set['item_id'])
    #添加边的标签
    label=triple_set['label']
    g.edata['label']=torch.LongTensor(triple_set['label'].values)

    return g

def eval(score, label):
    acc = accuracy_score(score, label)
    return acc