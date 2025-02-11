import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 데이터 X와 y를 생성하고, 학습용 데이터와 테스트용 데이터로 분리하여 반환하는 함수입니다.
def load_data():
    
    np.random.seed(0)
    
    X = 5*np.random.rand(100,1)
    y = 3*X + 5*np.random.rand(100,1)
    
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3, random_state=0)
    
    return train_X, train_y, test_X, test_y

# 회귀 모델을 불러오고, 불러온 모델을 학습용 데이터에 맞춰 학습시켜 반환하는 함수입니다.
def Linear_Regression(train_X, train_y):
    
    lr = LinearRegression()
    
    lr.fit(train_X,train_y)
    
    return lr
    
# 그래프로 시각화합니다.
def plotting_graph(test_X, test_y, predicted):
    plt.scatter(test_X,test_y)
    plt.plot(test_X, predicted, color='r')
    
    plt.savefig("MSE,MAE.png")

def main():
    
    train_X, train_y, test_X, test_y = load_data()
    
    lr =  Linear_Regression(train_X, train_y)
    
    predicted = lr.predict(test_X)
    
    MAE = mean_absolute_error(test_y, predicted)
    MSE = mean_squared_error(test_y, predicted)
    
    print("> MSE :",MSE)
    print("> MAE :",MAE)
    
    plotting_graph(test_X, test_y, predicted)
    
    return MSE, MAE

if __name__=="__main__":
    main()