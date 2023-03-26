#### 1. 加载数据：样本数据(dataloader), 知识图谱(dgl graph)
#### 2. 定义一下模型，传一些模型需要的参数，比如GCN模型需要图节点数目，定义优化器
#### 3. 训练模型
#### 4. 评估模型
#### 5. 节点的embedding保存起来
from config import *  ### 定义的参数
from model import GAT  #### 模型
#from preprocess_0313 import process_knowledge_graph,process_dataset
from dataloader import DiabetesDataset,get_dataloader
from util import construst_kg, eval
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# load_('') ### 加载镜像
#df=pd.read_csv('train_data_csv',columns=['user_id', 'item_id', 'lable'])
### 这里加在triple.tsv user id item id, relation
#g=dgl.ndata['feat']=torch.tensor((df['user_id'],df['item_id']))
### 1. 对现有特征做一些优化，特征工程，特征组合这些。2. 特征引入，引入新的特征
### 数据的优化工作收益一般大于模型优化
#1.加载数据集
datapath='dataset/KG/train_data.csv'
testpath='dataset/KG/test_data.csv'
train_loader=get_dataloader(datapath,batch_size=32,shuffle=True)
test_loader=get_dataloader(testpath,batch_size=32,shuffle=True)
#2.加载知识图谱
g=construst_kg()
### 2. bad case, 优化的重点方向
#g.edata['label']=torch.tensor(df['label'].values)
#3.定义模型
model = GAT(g.num_nodes(), in_feats, hid_feats, num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fcn = nn.CrossEntropyLoss()
#4.训练模型
for epoch in range(EPOCH):
    for data in train_loader:
        user_id = data['user_id']
        item_id = data['item_id']
        label = data['label']
        loss, scores, label=model(g, user_id, item_id)
        # labels=g.edata['label'].values
        # loss = F.cross_entropy(logits, labels)
        acc = eval(scores, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f, ACC: %.4f' % (epoch, loss.item(), acc))
    
#6.存储模型
torch.save(model.state_dict(), model_path)

#### 存储模型

