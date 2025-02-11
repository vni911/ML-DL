import numpy as np
import pandas as pd

name = ['하하', '김범수', '다현', '아이유', '최민식', '김혜수']
job  = ['가수', '가수'  , '가수', '가수'  , '배우'  , '배우']
height = [171, 182, 158, 160, 177, 170]
sex = ['M', 'M', 'F', 'F', 'M', 'F']

num = 0

node_list = {}

data = pd.DataFrame({'이름': name, '직업': job, '키': height,'성별': sex})
print(data,'\n')


def Sex_Node(df, depth):
    global num
    global node_list

    num +=1
    print('Node_num : {} | Node Depth : {} | Sex_Node'.format(num, depth))
    node_list[num] = 'Sex_Node'
    
    male = []
    female = []
    for idx, sex in enumerate(df['성별']):
        if sex == 'M':
            male.append(idx)
        elif sex == "F":
            female.append(idx)
    
    print('남자 Index : ',male)
    print('여자 Index : ',female)
    
    Job_Node(df, male , depth+1)
    Job_Node(df, female, depth+1)

def Job_Node(df,idx, depth):
    global num
    global node_list
    num +=1
    
    print('Node_num : {} | Node Depth : {} | Job_Node'.format(num, depth))
    node_list[num] = 'Job_Node'
    
    singer = []
    
    for i in idx:
        if df['직업'][i]=='가수':
            singer.append(i)
        else:
            num += 1
            print('Node_num : {} | Node Depth : {} | Name : {}'.format(num, depth+1 ,data['이름'][i]))
            node_list[num] = data['이름'][i]
    
    print('가수 Index : ',singer)
    
    Height_Node(df, singer, depth+1)

def Height_Node(df,idx, depth):
    global num
    global node_list
    num +=1
    print('Node_num : {} | Node Depth : {} | Height_Node'.format(num, depth))
    node_list[num] = 'Height_Node'
    
    for i in idx:
        num +=1
        if df['성별'][i] == 'M':
            if df['키'][i] < 180:
                print('Node_num : {} | Node Depth : {} | Name : {}'.format(num, depth+1,data['이름'][i]))
                node_list[num] = data['이름'][i]
            else:
                print('Node_num : {} | Node Depth : {} | Name : {}'.format(num, depth+1,data['이름'][i]))
                node_list[num] = data['이름'][i]
        else:
            if df['키'][i] < 160:
                print('Node_num : {} | Node Depth : {} | Name : {}'.format(num, depth+1,data['이름'][i]))
                node_list[num] = data['이름'][i]
            # 키가 160보다 큰 경우
            else:
                print('Node_num : {} | Node Depth : {} | Name : {}'.format(num, depth+1,data['이름'][i]))
                node_list[num] = data['이름'][i]

def main():
    Sex_Node(data, 1)
    print(node_list)

if __name__=="__main__":
    main()