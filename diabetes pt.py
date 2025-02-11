import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data():
    X, y = load_diabetes(return_X_y = True)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 100)
    
    return train_X, test_X, train_y, test_y

def reg_model(train_X, test_X, train_y):
    train_X, test_X, train_y, test_y = load_data()

    lr = LinearRegression()

    lr.fit(train_X, train_y)

    pred = lr.predict(test_X)


    return pred

def r_square(pred, test_y):
    train_X, test_X, train_y, test_y = load_data()

    lr = LinearRegression()

    lr.fit(train_X, train_y)

    pred = lr.predict(test_X)
    r2 = r2_score(test_y, pred)
    
    return r2
    
def main():
    train_X, train_y, test_X, test_y = load_data()

    r2  = r_square(reg_model(train_X, test_X, train_y), test_y)
    
    print("r2 score : ",r2)
    
    
if __name__ == "__main__":
    main()