#### 数据预处理模块，在公司里这个模块一般不需要自己写。。。处理逻辑需要掌握

### 1. 读取csv文件，(userid, itemid, category, feedback, timestamp)
### 2. 生成实体表，关系表，三元组表
### 3. 写入到文件 路径：dataset/KG/
import csv
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
def process_knowledge_graph(data_path='dataset/user_behavior.csv'):
    ### 实体：user,item
    entities_dict = dict() ### {entity_name : entity_id}, 实体集合表
    relations_dict = dict()
    triple_set = set()
    with open(data_path, 'r') as f:
        ### 假如有抬头
        for idx, line in enumerate(f):
            infos = line.split(',') ### 0: userid, 1: itemid ....
            user_id = 'user:' + infos[0]
            item_id = 'item:' + infos[1]
            action_id='feedback:'+infos[3]
            entities_dict[user_id]=item_id
            my_dict={'pv':1,'cart':2,'fav':3,'buy':5,'p':1}
            relations_dict[action_id]=my_dict[action_id]
            triple_set.add((user_id,item_id,action_id))
        triple_set=pd.DataFrame(np.mat(triple_set))
        headers=['user_id','item_id','action_id']
        triple_set.to_csv('triple.csv',header=headers,index=0)
        dict2csv(entities_dict,'dataset/KG/entities.csv')
        dict2csv(relations_dict,'dataset/KG/relation.csv')


def dict2csv(dic,filename):
    file=open(filename,'w',encoding='utf-8',newline='')
    csv_writer=csv.DictWriter(file,fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):
        dic1={key:dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
        file.close()




### 1. csv文件里面获取数据集，12月3日的数据(userid, itemid, category, feedback, timestamp)
### 2. 取有点击的数据，存成一个set表（userid, itemid, 1）。pv标识
### 3. 随机负采样，作为负样本，（userid, itemid, 0）,itemid 来自于全局的item
### 4. 把正负样本数据整合成一个set，做一个8:2的随机分割，分别存储到训练集和测试集文件 dataset/data
def process_dataset(data_path='dataset/user_behavior.csv'):
    df=pd.read_csv(data_path)
    df.columns=['userid', 'itemid', 'category', 'feedback', 'timestamp']
    triple1_set=set()
    triple2_set=set()
    df.loc[:,'timestamp']=df['timestamp'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
    df.loc[:,'date']=df['timestamp'].apply(lambda x:x.split(' ')[0])
    df.loc[:,'time']=df['timestamp'].apply(lambda x:x.split(' ')[1])
    test_df=df[(df["date"]='2017-12-03')&df["feedback"]='pv']
    for index,row in test_df.iterrows():
        triple1_set.add((row["userid"],row["itemid"],1))
    triple3_set=triple1_set.union(triple2_set)
    triple3_set=pd.DataFrame(np.mat(triple3_set))
    headers=['user_id','item_id','action_id']
    all_data=triple3_set.to_csv('triple.csv',header=headers,index=0)
    train_data,test_data=train_test_split(all_data,train_size=0.8,test_size=0.2)
    train_data.to_csv('dataset/data/train_data.csv')
    test_data.to_csv('dataset/data/test_data.csv')
    


if __name__=='main':
    ### 处理知识图谱
    process_knowledge_graph('')
