#Chapter 3, Example 2: Classifying Movie Reviews 
#Taweh Beysolow II 

#Import the necessary modules
import os, math
import numpy as np
import pandas as pan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Parameters 
np.random.seed(2018); n_estimators=1000
max_depth=10; learning_rate=1e-4; alpha=0.5

def summary_statistics(array):
    Min = min(array); Max = max(array); Range = Max - Min 
    Mean = np.mean(array); Sdev = np.std(array)
    output = pan.DataFrame([Min, Max, Range, Mean, Sdev]).T
    output.columns = ['Mean', 'Max', 'Range', 'Mean', 'SDev']
    return output
    
def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])
            
def load_data():
    negative_review_strings = os.listdir('/Users/tawehbeysolow/Downloads/review_data/tokens/neg')
    positive_review_strings = os.listdir('/Users/tawehbeysolow/Downloads/review_data/tokens/pos')
    negative_reviews, positive_reviews = [], []
    
    for positive_review in positive_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/pos/'+str(positive_review), 'r') as positive_file:
            positive_reviews.append(remove_non_ascii(positive_file.read()))
    
    for negative_review in negative_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/neg/'+str(negative_review), 'r') as negative_file:
            negative_reviews.append(remove_non_ascii(negative_file.read()))
    
    negative_labels, positive_labels = np.repeat(0, len(negative_reviews)), np.repeat(1, len(positive_reviews))
    labels = np.concatenate([negative_labels, positive_labels])
    reviews = np.concatenate([negative_reviews, positive_reviews])
    rows = np.random.random_integers(0, len(reviews)-1, len(reviews)-1)
    return reviews[rows], labels[rows]

def train_logistic_model(penalty, trials=1):
    x, y = load_data()
    t = TfidfVectorizer(min_df=10, max_df=300, stop_words='english', token_pattern=r'\w+')
    x = t.fit_transform(x).todense()
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end] , y[0:train_end]
    test_x, test_y = x[train_end:] , y[train_end:]
    auc_scores, accuracy_scores = [], []
    model = LogisticRegression(penalty=penalty)

    #Fitting and evaluating models
    for i in range(trials):
        model.fit(train_x, train_y)
        predicted_y_values = model.predict(train_x)
        accuracy_scores.append(accuracy_score(train_y, predicted_y_values))
        fpr, tpr = roc_curve(train_y, predicted_y_values)[0], roc_curve(train_y, predicted_y_values)[1]
        auc_scores.append(auc(fpr, tpr))
        
    #Evaluating training set and test set results 
    print('Summary Statistics from Training Set (AUC): \n' + str(summary_statistics(auc_scores)))
    print('Summary Statistics from Training Set (Accuracy): \n' + str(summary_statistics(accuracy_scores)))
    print('Training Data Confusion Matrix: \n' + str(confusion_matrix(train_y, predicted_y_values)))
    
    print('Summary Statistics from Test Set (AUC): \n' + str(summary_statistics(auc_scores)))
    print('Summary Statistics from Test Set (Accuracy): \n ' + str(summary_statistics(accuracy_scores)))
    print('Test Data Confusion Matrix: \n' + str(confusion_matrix(train_y, predicted_y_values)))
    
    #Plotting roc curve 
    predicted_y_values = model.predict(test_x)
    false_positive_rate, true_positive_rate = roc_curve(test_y, predicted_y_values)[0], roc_curve(test_y, predicted_y_values)[1]
    auc_score = auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' %auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for\n Logistic Regression ')
    plt.legend(loc="lower right")
    plt.show()
    
def train_models(trials=1):
    #Load and preprocess text data 
    x, y = load_data()
    t = TfidfVectorizer(min_df=10, max_df=300, stop_words='english', token_pattern=r'\w+')
    x = t.fit_transform(x).todense()
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end] , y[0:train_end]
    test_x, test_y = x[train_end:] , y[train_end:]
    r_auc_scores, r_accuracy_scores, b_auc_scores, b_accuracy_scores = [], [], [], []
    r = RandomForestClassifier(warm_start=True, max_depth=max_depth, n_estimators=n_estimators)
    b = BernoulliNB(alpha=alpha)
    
    #Fitting and evaluating models
    for i in range(trials):
        if i%10 == 0 and i > 0:
            print('Trial ' + str(i) + ' out of 1 completed')
        r.fit(train_x, train_y), b.fit(train_x, train_y)
        r_predicted_y_values = r.predict(train_x)
        r_accuracy_scores.append(accuracy_score(train_y, r_predicted_y_values))
        fpr, tpr = roc_curve(train_y, r_predicted_y_values)[0], roc_curve(train_y, r_predicted_y_values)[1]
        r_auc_scores.append(auc(fpr, tpr))
        r_accuracy_scores.append(accuracy_score(train_y, r_predicted_y_values))
        
        b_predicted_y_values = b.predict(train_x)
        b_accuracy_scores.append(accuracy_score(train_y, b_predicted_y_values))
        fpr, tpr = roc_curve(train_y, b_predicted_y_values)[0], roc_curve(train_y, b_predicted_y_values)[1]
        b_auc_scores.append(auc(fpr, tpr))
        
    #Evaluating training set and test set results 
    print('Summary Statistics from Training Set Random Forest (AUC): \n' + str(summary_statistics(r_auc_scores)))
    print('Summary Statistics from Training Set Random Forest (Accuracy): \n' + str(summary_statistics(r_accuracy_scores)))
    print('Training Data Confusion Matrix (Random Forest): \n' + str(confusion_matrix(train_y, r_predicted_y_values)))
    
    print('Summary Statistics from Training Set Naive Bayes (AUC): \n' + str(summary_statistics(b_auc_scores)))
    print('Summary Statistics from Training Set Naive Bayes (Accuracy): \n' + str(summary_statistics(b_accuracy_scores)))
    print('Training Data Confusion Matrix (Naive Bayes): \n' + str(confusion_matrix(train_y, b_predicted_y_values)))
    
    if np.mean(r_auc_scores) > np.mean(b_auc_scores): model = r
    else: model = b
    
    if model == b: model_string = 'Naive Bayes Classifier'
    else: model_string = 'Random Forest'
    
    #Evaluating best model and plotting roc curve 
    predicted_y_values = model.predict(test_x)
    false_positive_rate, true_positive_rate = roc_curve(test_y, predicted_y_values)[0], roc_curve(test_y, predicted_y_values)[1]
    auc_score = auc(false_positive_rate, true_positive_rate)
    print('Test Data Confusion Matrix: \n' + str(confusion_matrix(test_y, predicted_y_values)))
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' %auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for\n ' + str(model_string))
    plt.legend(loc="lower right")
    plt.show()
        
if __name__ == '__main__': 
    
    #train_logistic_model(penalty='l1')
    
    #train_logistic_model(penalty='l2')
    
    train_models()
    
    
        