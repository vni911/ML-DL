from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_data():
    X, y = load_breast_cancer(return_X_y = True)
    train_X, test_X, train_y ,test_y = train_test_split(X, y, test_size = 0.2, random_state = 156)
    return train_X, test_X, train_y ,test_y

def Random_clf(train_X, train_y, test_X):
    rfc = RandomForestClassifier(n_estimators = 100, max_depth = 30, min_samples_leaf = 2, min_samples_split = 3,random_state = 100)
    rfc.fit(train_X, train_y)
    pred = rfc.predict(test_X)
    return rfc, pred
    
def main():
    train_X, test_X, train_y ,test_y = load_data()
    rfc, pred = Random_clf(train_X, train_y, test_X)
    print('테스트 데이터 예측 정확도 : {0:.4f}'.format(accuracy_score(test_y, pred)))
    
if __name__ == "__main__":
    main()
