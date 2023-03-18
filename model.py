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
    def __init__(self,in_feats,hid_feats,num_heads) -> None:
        super(GAT,self).__init__()
        self.conv1=dglnn.GATConv(in_feats,hid_feats,num_heads)
        self.conv2=dglnn.GATConv(hid_feats*num_heads,hid_feats,1)
    def forward(self,g,feats):
        feats=self.conv1(g,feats).flatten(1)
        feats=F.relu(feats)
        feats=self.conv2(g,feats).mean(1)
        return feats
    def train(g,model,labels):
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


