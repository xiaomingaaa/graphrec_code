### 推理，和训练流程差不多
from config import *  ### 定义的参数
from model import GCN,GAT  #### 模型
#from preprocess_0313 import process_knowledge_graph,process_dataset
from dataloader import DiabetesDataset,get_dataloader
from util import construst_kg, eval
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


g=construst_kg()
model = GAT(g.num_nodes(), in_feats, hid_feats, num_heads)
model.load_state_dict(model_path)
with torch.no_grad:
    user_id = 2
    item_id = [12,13,,,,,,,,,,,,]
    score = model.inference(g, user_id, item_id)
