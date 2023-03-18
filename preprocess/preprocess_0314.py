# 数据预处理模块，在公司里这个模块一般不需要自己写。。。处理逻辑需要掌握

# 1. 读取csv文件，(userid, itemid, category, feedback, timestamp)
# 2. 生成实体表，关系表，三元组表
# 3. 写入到文件 路径：dataset/KG/
'''
user:10001,0
user:10003,1

item:20001,2

x = [
    [1,2,3],
    [1,3,4],
    [2,0,0]
]
y = f(x)

y=[
    [1,2,3],
    [1,3,4],
    [2,0,0]
]

index = [0,1,2]
y[index] ->
'''
import csv
import pandas as pd
import numpy as np
import time
#from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import random


def process_knowledge_graph(data_path='dataset/UserBehavior.csv'):
    # 实体类型：user,item
    entities_dict = dict()  # {entity_name : entity_id}, 实体集合表
    relations_dict = dict()
    triple_set = set()
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        # 假如有抬头
        index = 0
        relation_index = 0
        for idx, line in enumerate(f):
            infos = line.split(',')  # 0: userid, 1: itemid ....
            user_id = 'user:' + infos[0]
            item_id = 'item:' + infos[1]
            action_id = 'feedback:'+infos[3]
            timestamp = int(infos[4])
            if timestamp>= 1512230400:
                continue
            # entities_dict[user_id]=item_id ###
            # entities_dict['user:10001']=0 ###
            if user_id not in entities_dict:
                entities_dict[user_id] = index
                index += 1
            if item_id not in entities_dict:
                entities_dict[item_id] = index
                index += 1
            # entities_dict[user_id] = len(entities_dict)
            # entities_dict[item_id] = len(entities_dict)
            # 关系编码
            if action_id not in relations_dict:
                relations_dict[action_id] = relation_index
                relation_index += 1
            triple_set.add((entities_dict[user_id], entities_dict[item_id], relations_dict[action_id]))
            # my_dict = {'pv': 1, 'cart': 2, 'fav': 3, 'buy': 5, 'p': 1}
            # relations_dict[action_id] = my_dict[action_id]
    with open('dataset/kg/entities.tsv', 'w') as f:
        for entity_name, entity_id in entities_dict.items():
            f.write('{}\t{}\n'.format(entity_name, entity_id))
    with open('dataset/kg/relations.tsv', 'w') as f:
        for k, v in relations_dict.items():
            f.write('{}\t{}\n'.format(k, v))
    with open('dataset/kg/triple.tsv', 'w') as f:
        for row in triple_set:
            f.write('{}\t{}\t{}\n'.format(row[0], row[1], row[2]))


#         """ triple_set=pd.DataFrame(np.mat(triple_set))
#         headers=['user_id','item_id','action_id']
#         triple_set.to_csv('triple.csv',header=headers,index=0)
#         dict2csv(entities_dict,'dataset/KG/entities.csv')
#         dict2csv(relations_dict,'dataset/KG/relation.csv')

#  """
# """ def dict2csv(dic,filename):
#     file=open(filename,'w',encoding='utf-8',newline='')
#     csv_writer=csv.DictWriter(file,fieldnames=list(dic.keys()))
#     csv_writer.writeheader()
#     for i in range(len(dic[list(dic.keys())[0]])):
#         dic1={key:dic[key][i] for key in dic.keys()}
#         csv_writer.writerow(dic1)
#         file.close() """


# 1. csv文件里面获取数据集，12月3日的数据(userid, itemid, category, feedback, timestamp)
# 2. 取有点击的数据，存成一个set表（userid, itemid, 1）。pv标识
# 3. 随机负采样，作为负样本，（userid, itemid, 0）,itemid 来自于全局的item
# 4. 把正负样本数据整合成一个set，做一个8:2的随机分割，分别存储到训练集和测试集文件 dataset/data

def process_dataset(data_path='dataset/UserBehavior.csv'):
    item_set = set()
    positive_triple_set1=set()
    dec_03_data =list()
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for idx ,line in enumerate(f):
            #user_id, item_id, category_id, behavior_type, timestamp = row.split(',')
            infos = line.strip('\n').split(',')  # 0: userid, 1: itemid ....
            print(idx,infos)
            user_id = 'user:' + infos[0]
            item_id = 'item:' + infos[1]
            action_id = 'feedback:'+infos[3]
            timestamp = int(infos[4])
            item_set.add(item_id)
            #datetime_obj = datetime.strptime(timestamp, '%Y-%m-%d %H')
        #dec_03_data = [row for row in enumerate(f) if row[4].startwith('2017-12-03') & row[3] == 'pv']
            if timestamp>=1512230400:
                # dec_03_data.append([infos[0],infos[1],infos[3]])
                dec_03_data.append([user_id,item_id,infos[3]])
            # if idx>100:
            #     break
        # print(dec_03_data)
            
        #dec_03_data = [row for row in enumerate(f) if row[4]>=1512230400& row[3] == 'pv']
    # 使用编码之后的id
    entity_dict = dict()
    with open('dataset/kg/entities.tsv', 'r') as f:
        for line in f:
            infos = line.strip('\n').split('\t') 
            # print(infos)
            entity_dict[infos[0]] = int(infos[1])
    print(entity_dict)
    for user_id, item_id, action_id in dec_03_data:
        # infos=line.split(',')
        # user_id, item_id, action_id = infos[0], infos[1], infos[2]
        if action_id=='pv':
            positive_triple_set1.add((entity_dict[user_id], entity_dict[item_id], 1))
    
    positive_triple_set=set(positive_triple_set1)
    #positive_triple_set1 = [(entity_dict[user_id], entity_dict[item_id], 1)for row in dec_03_data]
    #positive_triple_set=set(positive_triple_set1)
    negtive_triple_set = set()
    for user_id, item_id, action_id in dec_03_data:
        # infos=line.split(',')
        # user_id, item_id, action_id = infos[0], infos[1], infos[2]
        #user_id, item_id, category_id, behavior_type, timestamp = row.split(',')
        #### 修改一下生成的负样本数量 1：500
        for _ in range(1):
            neg_items_id = random.choice(list(item_set))
            if(entity_dict[user_id], entity_dict[neg_items_id], 1) not in positive_triple_set:
                negtive_triple_set.add((entity_dict[user_id], entity_dict[neg_items_id], 0))

    all_data_set = positive_triple_set.union(negtive_triple_set)
    print(all_data_set)
    all_data_list = list(all_data_set)
    # 打乱正负样本数据集
    random.shuffle(all_data_list)
    train_size=int(len(all_data_list)*0.8)
    train_data=all_data_list[:train_size]
    test_data=all_data_list[train_size:]
    print(test_data)
    #train_data, test_data = train_test_split(list(all_data_set), train_size=0.8, test_size=0.2)
    # train_data.to_csv('dataset/data/train_data.csv')
    # test_data.to_csv('dataset/data/test_data.csv')
    pd.DataFrame(train_data,columns=['user_id', 'item_id', 'lable']).to_csv('dataset/KG/train_data.csv',index=False)
    pd.DataFrame(test_data,columns=['user_id', 'item_id', 'lable']).to_csv('dataset/KG/test_data.csv',index=False)

if __name__ == '__main__':
    # 处理知识图谱
    # process_knowledge_graph()
    process_dataset()

    # """ df=pd.read_csv(data_path)
    # df.columns=['userid', 'itemid', 'category', 'feedback', 'timestamp']
    # triple1_set=set()
    # triple2_set=set()
    # df.loc[:,'timestamp']=df['timestamp'].apply(
    #     lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
    # df.loc[:,'date']=df['timestamp'].apply(lambda x:x.split(' ')[0])
    # df.loc[:,'time']=df['timestamp'].apply(lambda x:x.split(' ')[1])
    # test_df=df[(df["date"]='2017-12-03')&df["feedback"]='pv']

    # for index,row in test_df.iterrows():
    #     triple1_set.add((row["userid"],row["itemid"],1))

    # triple3_set=triple1_set.union(triple2_set)
    # triple3_set=pd.DataFrame(np.mat(triple3_set))
    # headers=['user_id','item_id','action_id']
    # all_data=triple3_set.to_csv('triple.csv',header=headers,index=0)
    # train_data,test_data=train_test_split(all_data,train_size=0.8,test_size=0.2)
    # train_data.to_csv('dataset/data/train_data.csv')
    # test_data.to_csv('dataset/data/test_data.csv')
    #  """


