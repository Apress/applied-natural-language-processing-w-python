#Text classification demo
#Taweh Beysolow II 

#Import the necessary modules
import math 
import numpy as np
import pandas as pan
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Parameters
np.random.seed(2018)
trials=100
n_estimators=1000
max_depth=20
learning_rate=1e-4

def summary_statistics(array):
    Min = min(array)
    Max = max(array)
    Mean = np.mean(array)
    Sdev = np.std(array)
    Range = Max - Min
    output = pan.DataFrame([Min, Max, Mean, Sdev, Range]).T
    output.columns = ['Min', 'Max', 'Mean', 'SDev', 'Range']
    return output

def explore_data(message_length=True):
    l = LabelEncoder()
    data = pan.read_csv('/Users/tawehbeysolow/Downloads/smsspamcollection/SMSSPamCollection.csv',
                        delimiter='\t', 
                        header=None)
    
    if message_length==True:
        tokenizer = RegexpTokenizer(r'\w+')
        length_count = [len(tokenizer.tokenize(data[1][i])) for i in range(0, len(data))]
        plt.hist(length_count)
        plt.title('Histogram of SMS Message Length')
        plt.xlabel('Message Length in Words')
        plt.ylabel('Relative Frequency')
    
    else:
        labels = l.fit_transform(data[0].values)
        plt.hist(labels)
        plt.title('Histogram of Class Labels')
        plt.xlabel('Message Length in Words')
        plt.ylabel('Relative Frequency')
        
    return None

#explore_data(message_length=False)
    
def load_spam_data():
    c = CountVectorizer(stop_words='english', token_pattern=r'\w+')
    l = LabelEncoder()
    data = pan.read_csv('/Users/tawehbeysolow/Downloads/smsspamcollection/SMSSPamCollection.csv',
                        delimiter='\t', 
                        header=None)
    print(data.head())
    x = c.fit_transform(data[1]).todense()
    y = l.fit_transform(data[0])
    print('Vocabulary Size: ' + str(len(c.vocabulary_)))
    return x, y 

def train_logistic_model(penalty='l1'):
    '''
    Training random forest model using simple BOW model on text data
    '''
    x, y = load_spam_data()
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end, :], y[0:train_end]
    test_x, test_y = x[train_end:, :], y[train_end:]

    #Fitting training algorithm 
    l = LogisticRegression(penalty=penalty)
    accuracy_scores, auc_scores = [], []

    for i in range(trials):
        if i%10 == 0 and i > 0:
            print('Trial ' + str(i) + ' out of 100 completed')
        l.fit(train_x, train_y)
        predicted_y_values = l.predict(train_x)
        accuracy_scores.append(accuracy_score(train_y, predicted_y_values))
        fpr, tpr = roc_curve(train_y, predicted_y_values)[0], roc_curve(train_y, predicted_y_values)[1]
        auc_scores.append(auc(fpr, tpr))
             
    #Evaluating training set results
    print('Summary Statistics (AUC): \n' + str(summary_statistics(auc_scores)))
    print('\nSummary Statistics (Accuracy Scores): \n' + str(summary_statistics(accuracy_scores)))
    
    #Plotting test model results
    predicted_y_values = l.predict(test_x)
    print('\nTest Model Accuracy: ' + str(accuracy_score(test_y, predicted_y_values)))
    false_positive_rate, true_positive_rate = roc_curve(test_y, predicted_y_values)[0], roc_curve(test_y, predicted_y_values)[1]
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('\nTest True Positive Rate: ' + str(true_positive_rate[1]) + '\n')
    print('\nTest False Positive Rate: ' + str(false_positive_rate[1]) + '\n')
    print(confusion_matrix(test_y, predicted_y_values))
    
    #Plotting roc curve 
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for\n Logistic Regression ')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == '__main__': 
    
    train_logistic_model(penalty='l1')
       