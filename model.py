#### 模型代码
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self,in_feats,hid_size,out_feats,num_nodes) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.embedding = nn.Embedding(num_nodes, in_feats)
        ### GCN
        self.layers.append(dglnn.GraphConv(in_feats,hid_size,activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size,out_feats))
        self.dropout=nn.Dropout(0.5)
        

    def forward(self,g):
        h=g.ndata['feat'] ### ids 1 x n, 0,1,2,3
        h = self.embedding[h] ### n x in_feats, 0,1,2,3
        for i,layer in enumerate(self.layers):
            if i!=0:
                h=self.dropout(h)
            h=layer(g,h)
        return h
def train(g,labels,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fcn = nn.CrossEntropyLoss()
    for epoch in range(200):
        logits=model(g)
        labels=g.edata['label'].values
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


# model = GCN(num_node=100)
# h = model(h)

class GAT(nn.Module):
    def __init__(self,num_nodes,in_feats,hid_feats,num_heads) -> None:
        super(GAT,self).__init__()
        self.embedding = nn.Embedding(num_nodes, in_feats)
        self.conv1=dglnn.GATConv(in_feats,hid_feats,num_heads)
        self.conv2=dglnn.GATConv(hid_feats*num_heads,hid_feats,1)
    
    def forward(self, g, user_ids, item_ids):
        ### user id, item id: batch
        feats = self.embedding[g.ndata['feat']]
        ### num_nodes x in_feats
        feats=self.conv1(g,feats).flatten(1)
        feats=F.relu(feats)
        feats=self.conv2(g,feats).mean(1)
        ### num_nodes x hid_feats
        user_embed = feats[user_ids]
        ## 32 x hid_feats
        item_embed = feats[item_ids]
        ## 32 x hid_feats
        ### 32 x hid_feats * hid_feats x 32 = 32 x 32
        '''
        user1   item1 1
        user2   item2 1
        user_embed = [
            [1,2,,,,,,,,,,,,]
            [2,1,,,,,,,,,,,,]
        ]

        item_embed = [
            [1,3,,,,,,,,,,,,]
            [2,1,,,,,,,,,,,,]
        ] 
        -> [
            [1,2]
            [3,1]
        ]

        scores = [
            [7, 4]
            [5, 5]
        ]
        '''
        scores = torch.multiply(user_embed, item_embed)
        labels = torch.ones(user_embed.shape[0])
        labels = torch.diag(labels)
        ### 分类
        loss = F.cross_entropy(scores, labels)
        return loss, scores, labels
    
    def inference(self, g, user_id, item_id):
        feats = self.embedding[g.ndata['feat']]
        ### num_nodes x in_feats
        feats=self.conv1(g,feats).flatten(1)
        feats=F.relu(feats)
        feats=self.conv2(g,feats).mean(1)
        ### num_nodes x hid_feats
        user_embed = feats[user_ids]
        ## 32 x hid_feats
        item_embed = feats[item_ids]
        scores = torch.multiply(user_embed, item_embed)
        return scores

# class Trainer(nn.Module):
#     def __init__(self,num_nodes,in_feats,hid_feats,num_heads) -> None:
#         super(GAT,self).__init__()
#         self.embedding = nn.Embedding(num_nodes, in_feats)
#         self.conv1=dglnn.GATConv(in_feats,hid_feats,num_heads)
#         self.conv2=dglnn.GATConv(hid_feats*num_heads,hid_feats,1)
    
#     def forward(self, g, user_ids, item_ids):
#         ### user id, item id: batch
#         feats=self.conv1(g,feats).flatten(1)
#         feats=F.relu(feats)
#         feats=self.conv2(g,feats).mean(1)
#         return feats
    # def train(g,model,labels):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #     loss_fcn = nn.CrossEntropyLoss()
    #     for epoch in range(200):
    #         logits=model(g)
    #         labels=g.edata['label'].values
    #         loss = F.cross_entropy(logits, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


