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
    g = dgl.DGLGraph()
    triple_set=pd.read_csv('dataset/KG/triple.tsv',header=None, delimiter='\t')
    entities_dict=pd.read_csv('dataset/KG/entities.tsv',header=None, delimiter='\t')
    #构建dgl图
    g.add_nodes(len(entities_dict))
    g.ndata['feat']=torch.LongTensor(entities_dict.iloc[:,1])
    #添加边到dgl图
    g.add_edges(triple_set.loc[:,0],triple_set.iloc[:,1])
    #添加边的标签
    label=triple_set.iloc[:,2]
    g.edata['label']=torch.LongTensor(triple_set.iloc[:,2].values)

    g = dgl.add_self_loop(g)

    return g

def eval(score, label):
    acc = accuracy_score(score, label)
    return acc

if __name__ == '__main__':
    G=construst_kg()
    y_true = G.edata['label'].numpy()
    y_pred = torch.randint(0, 2, (len(y_true),)).numpy()
    # 评估
    acc = eval(y_pred, y_true)
    print('Accuracy: {:.4f}'.format(acc))
    