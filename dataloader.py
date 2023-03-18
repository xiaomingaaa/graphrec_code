### 生成用于模型的数据集
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DiabetesDataset(Dataset):
    def __init__(self,datapath):
        self.data=pd.read_csv(datapath, columns=['user_id', 'item_id', 'lable'])
        #self.len=df.columns[0] ### 行数
        self.user_ids = torch.LongTensor(self.data['user_id'].values)
        self.item_ids = torch.LongTensor(self.data['item_id'].values)
        self.labels = torch.FloatTensor(self.data['label'].values)
        #self.y_data=df['label'] ### 保证label和user id, item id为数值

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]
        label = self.labels[index]
        #return torch.tensor(self.x_data[index]),torch.tensor(self.y_data[index])
        return {'user_id': user_id, 'item_id': item_id, 'label': label}
    

def get_dataloader(datapath,batch_size,shuffle=True):
    dataset=DiabetesDataset(datapath)
    dataloader=DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle)
    return dataloader

if __name__=='__main__':
    pass

#dataset1=DiabetesDataset('train_data_csv')
#train_loader=DataLoader(dataset=dataset1,
                        #batch_size=32,
                        #shuffle=True,
                        #num_workers=2)
#dataset2=DiabetesDataset('test_data_csv')
#test_loader=DataLoader(dataset=dataset2,
                       #batch_size=32,
                       #shuffle=True,
                       #num_workers=2)

### odps  数据平台，数据读取的代码
### 星云，模型运行平台