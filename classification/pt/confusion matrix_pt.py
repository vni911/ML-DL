from sklearn.model_selection import train_test_split
from draw_image import draw_digit_images
from sklearn.datasets import load_digits

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action = 'ignore')

def load_data():
    X, y = load_digits(return_X_y = True)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 100)
  
    
    return train_X, test_X, train_y, test_y
    
def SVM_clf(train_X, test_X, train_y):
    
    svm = SVC()
    svm.fit(train_X, train_y)
    pred = svm.predict(test_X)
    return pred

def cal_eval(test_y, pred):
    conf_matrix = confusion_matrix(test_y, pred)
    recall_scores = recall_score(test_y, pred, average=None) 
    precision_scores = precision_score(test_y, pred, average=None)
    accuracy = accuracy_score(test_y, pred)
    index_3_recall = recall_scores[3]
    index_3_precision = precision_scores[3]
    
    return index_3_precision, index_3_recall, accuracy

def main():
    
    train_X, test_X, train_y, test_y = load_data()   
    pred = SVM_clf(train_X, test_X, train_y)
    
    print("Confusion matrix results :\n\t- row : real(test_y) 0 ~ 9 label\n\t- column : predicted 0 ~ 9 label\n\n%s\n"  % confusion_matrix(test_y, pred))
    
    index_3_precision, index_3_recall, accuracy = cal_eval(test_y, pred)
    
    print("index 3의 recall : %f" % index_3_recall)
    print("index 3의 precision : %f" % index_3_precision)
    print("전체 accuracy : %f" % accuracy)

    draw_digit_images(test_X, test_y, pred)

if __name__ == "__main__":
    main()