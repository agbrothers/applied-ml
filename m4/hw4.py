# Generally, a parameter selection procedure might be necessary to evaluate Probability of
# Detection versus Probability of False Alarm (i.e., Pd versus Pf) in order to select a classifier 
# model and/or select a value for a hyperparameter for a classifier. 
 
# In this assignment we will produce an ROC plot presenting operating points of various 
# classifiers and their varying hyperparameters so that we can make a justifiable operating 
# classifier/parameter selection for the following problem. 
 
# The classification of fake news or misinformation is a very important task today. Download the 
# fake news dataset (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), 
# Fake.csv and True.csv files. Load the datasets into your model development framework and 
# examine the features to confirm that they are text in title and text columns. Set fake as 1 
# and true as 0. Concatenate the datasets together to produce one dataset of around 44,880 
# rows. Apply necessary pre-processing  to extract the title column with Tf-Idf. (This assigns 
# numerical values to terms based on their frequency in a given document and throughout a 
# given collection of documents.) Use around 50 features. Make sure to include a sanity check in 
# the pipeline and perhaps run your favorite baseline classifier first. 


import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, train_test_split    



## LOAD DATASET
df_true = pd.read_csv("m4/true.csv")
df_fake = pd.read_csv("m4/fake.csv")

## PREPROCESSING
df_true['class'] = 0; df_fake['class'] = 1 
df = pd.concat([df_fake, df_true], ignore_index=True) 
n = len(df)

## CLEANING
unique_title = 100 * df["title"].nunique() / df.title.count()
print(f"Unique Titles: {unique_title:.2f}%")

## DROP TITLE DUPLICATES
df = df.drop_duplicates('title')
print(f"Dropped {n-len(df)} title duplicates")
n = len(df)

## CHECK CLASS BALANCE AFTER REDUCING DATASET
df.reset_index(inplace=True)
class_balance = 100 * df["class"].sum() / df["class"].count()
print(f"Class balance: {class_balance:.2f}% Fake Samples")

## COMPUTE FEATURES AND TARGETS
X = TfidfVectorizer(stop_words='english', max_features=50).fit_transform(df['title'])
y = df["class"]
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)



## PROBLEM 1
import numpy as np 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import trange


def train(model, X, y, k=10):
    ## K-FOLD CROSS VALIDATION
    acc, tpr, fpr = [], [], []
    kfold = KFold(n_splits=k,shuffle=False)
    for i_train, i_test in kfold.split(X, y):
        
        ## TRAIN MODEL
        model.fit(X[i_train], y[i_train])
        y_pred = model.predict(X[i_test])

        ## COMPUTE METRICS
        conf = metrics.confusion_matrix(y[i_test], y_pred)
        tpr.append(conf[0,0] / (conf[0,0] + conf[1,0] + 1e-10))
        fpr.append(conf[0,1] / (conf[0,0] + conf[0,1] + 1e-10))
        acc.append(metrics.accuracy_score(y[i_test], y_pred))

    return np.mean(acc), np.mean(tpr), np.mean(fpr)


def plot_ROC(tpr, fpr, title):
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    ax.scatter(fpr, tpr, 5, color='red', marker='o', label='Operating Points')
    tpr = [0.] + list(tpr) + [1.]
    fpr = [0.] + list(fpr) + [1.]
    ax.plot(fpr, tpr, linestyle=':', label='ROC')
    ax.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Coin Flip')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")    
    plt.tight_layout()
    return


hyperparameters = 10

## DECISION TREE
acc, tpr, fpr = [], [], []
depth = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
n_feat = (3, 5, 7, 9, 11, 13, 15, 17, 19, 20)
for p in trange(hyperparameters):
    dtree = DecisionTreeClassifier(max_depth=depth[p], max_features=n_feat[p])
    a,t,f = train(dtree, X, y)
    acc.append(a); tpr.append(t); fpr.append(f)
print(f"Accuracy: {np.round(acc,2)}")
plot_ROC(tpr, fpr, title="ROC Decision Tree")

                         
## RANDOM FOREST
acc, tpr, fpr = [], [], []
n_est = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
for p in trange(hyperparameters):
    rf = RandomForestClassifier(n_estimators=n_est[p], max_features=n_feat[p])
    a,t,f = train(rf, X, y, k=3)
    acc.append(a); tpr.append(t); fpr.append(f)
print(f"Accuracy: {np.round(acc,2)}")
plot_ROC(tpr, fpr, title="ROC Random Forest")


## NEURAL NETWORK
acc, tpr, fpr = [], [], []
h = (16, 32, 64, 96, 128, 160, 256, 310, 384, 512)
for p in trange(hyperparameters):
    rf = MLPClassifier(hidden_layer_sizes=(h[p],h[p]))
    a,t,f = train(rf, X, y, k=3)
    acc.append(a); tpr.append(t); fpr.append(f)
print(f"Accuracy: {np.round(acc,2)}")
plot_ROC(tpr, fpr, title="ROC Neural Network")









## PROBLEM 4
import scipy

## CLEANING
unique_text = 100 * df["text"].nunique() / df.title.count()
print(f"Unique Text: {unique_text:.2f}%")

## DROP TEXT DUPLICATES
df = df.drop_duplicates('text')
df.reset_index(inplace=True)
print(f"Dropped {n-len(df)} remaining text duplicates")
n = len(df)

X_text = TfidfVectorizer(stop_words='english', max_features=50).fit_transform(df['text'])
X_title = TfidfVectorizer(stop_words='english', max_features=50).fit_transform(df['title'])
X = scipy.sparse.csr_matrix(scipy.sparse.hstack((X_title, X_text)))
y = df["class"]

## DECISION TREE
acc, tpr, fpr = [], [], []
depth = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
n_feat = (3, 5, 7, 9, 11, 13, 15, 17, 19, 20)
for p in trange(hyperparameters):
    dtree = DecisionTreeClassifier(max_depth=depth[p], max_features=n_feat[p])
    a,t,f = train(dtree, X, y)
    acc.append(a); tpr.append(t); fpr.append(f)
print(f"Accuracy: {np.round(acc,2)}")
plot_ROC(tpr, fpr, title="ROC Decision Tree")
