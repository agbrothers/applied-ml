## DATASET EXPLORATION
import numpy as np
import pandas as pd 
import seaborn as sns; 
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (4,6)
sns.set(style="ticks", color_codes=True) 

## DATA PIPELINE
df = pd.read_csv("m6/heart_dataset.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Number of Duplicate Samples: {df.duplicated().sum()}")

sample = df.iloc[0]
dtypes = df.dtypes
nans = 100 * df.isna().sum() / len(df)
categories = df.nunique()
print(pd.concat(
    (sample, dtypes, nans, categories), 
    keys=["Sample", "Datatypes", "NaN %", "Num Categories"], 
    axis=1)
)



## RE-TYPE CATEGORICAL VARIABLES
numeric = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
categorical = ["Sex","ChestPainType","FastingBS","RestingECG","ExerciseAngina","ST_Slope","HeartDisease"]
ordinal = []
print("Categorical Variables:")
for feat in categorical: 
    print(f"{feat}: {set(df[feat])}")
    df[feat] = df[feat].astype('category')


# ## CONVERT CATEGORICAL VARIABLES TO NUMERIC REPRESENTATIONS
# hist_df = df.copy()
# hist_df[categorical] = hist_df[categorical].apply(lambda x: x.cat.codes)
# print(pd.concat((hist_df.iloc[0], hist_df.dtypes, hist_df.max(), hist_df.min(), hist_df.std()), keys=["Sample", "Datatypes", "Max", "Min", "Std"], axis=1))

# # plt.clf()
# hist_df.hist(bins=50, figsize=(8,8), layout=(6,2))
# plt.tight_layout()

## ONE-HOT ENCODE CATEGORICALS
df = pd.get_dummies(df)

# ## EXAMINE CORRELATIONS BETWEEN FEATURES
# plt.clf()
# plt.rcParams["figure.figsize"] = (4,4)
# heatmap = sns.heatmap(df.corr(), cmap="bwr")
# heatmap.set_title("Correlation Matrix")
# plt.tight_layout()

## REMOVE OUTLIERS AND INVALID ROWS
print("Percentage of Invalid Samples:")
print(f"Cholesterol: {sum(df.Cholesterol == 0) / len(df):.3f}")
print(f"Oldpeak: {sum(df.Oldpeak < 0) / len(df):.3f}")
print(f"RestingBP: {sum(df.RestingBP < 90) / len(df):.3f}")
df.drop(df[df.Oldpeak < 0].index, inplace=True)
df.drop(df[df.RestingBP < 90].index, inplace=True)

from sklearn.impute import KNNImputer
df.Cholesterol.replace(0, np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# df[["Cholesterol","Oldpeak","RestingBP"]].hist(bins=50, figsize=(8,6), layout=(3,1))
# plt.tight_layout()

## MIN-MAX NORMALIZE DATA
df_norm = (df - df.min()) / (df.max()-df.min())




## 1. [10 pts] Report 10-fold cross-validation (“CV”) performances of the following types of classifiers, using default parameters:  
## • GaussianNB 
## • Linear SVC (use SVC(kernel='linear', probability=True)) 
## • MLPClassifier 
## • DecisionTreeClassifier 
## Now report the RandomForestClassifier performance too. Since this is already an ensemble classifier, this one does not need to be done with CV. 

## TRAIN MODEL
from sklearn.model_selection import KFold, train_test_split    
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
import numpy as np 

def train(X, y, model_class=None, model_config=None, model=None, k=10, shuffle=False, split=0.8):
    assert (model_class and model_config is not None) or model, \
        "Must provide an initialized model or a class and config"
    
    ## K-FOLD CROSS VALIDATION
    acc, conf = [], []
    if k > 1:
        kfold = KFold(n_splits=k, shuffle=shuffle)
        for i_train, i_test in kfold.split(X, y):
            
            ## TRAIN MODEL
            model = model_class(**model_config)
            model.fit(X[i_train], y[i_train])
            y_pred = model.predict(X[i_test])

            ## COMPUTE METRICS
            conf.append(metrics.confusion_matrix(y[i_test], y_pred))
            acc.append(metrics.accuracy_score(y[i_test], y_pred))
    ## SINGLE TRAIN-TEST SPLIT
    else:
        ## TRAIN MODEL
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=split)
        model = model or model_class(**model_config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ## COMPUTE METRICS
        conf.append(metrics.confusion_matrix(y_test, y_pred))
        acc.append(metrics.accuracy_score(y_test, y_pred))

    conf = np.array(conf)
    tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
    fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
    print(f"{model.__class__.__name__} {k}-fold Mean Accuracy: {100*np.mean(acc):.4} ± {100*np.std(acc):.4}%")
    print(f"    Mean True Positive Rate: {100*np.mean(tpr):.4}%")
    print(f"    Mean False Positive Rate: {100*np.mean(fpr):.4}%\n")
    return acc, conf


X = df_norm.drop(columns=["HeartDisease_0","HeartDisease_1"]).values
y = df_norm["HeartDisease_1"].values


## NAIVE BAYES
model_config = {}
acc, conf = train(X, y, GaussianNB, model_config, k=10, shuffle=True)

## SVM
model_config = {"kernel":"linear", "probability":True}
acc, conf = train(X, y, SVC, model_config, k=10, shuffle=True)
# model_config = {"dual":False}
# acc, conf = train(X, y, LinearSVC, model_config, k=10, shuffle=True)

## NEURAL NET
model_config = {}
acc, conf = train(X, y, MLPClassifier, model_config, k=10, shuffle=True)

## DECISION TREE
model_config = {}
acc, conf = train(X, y, DecisionTreeClassifier, model_config, k=10, shuffle=True)

## RANDOM FOREST
model_config = {}
acc, conf = train(X, y, RandomForestClassifier, model_config, k=1, shuffle=True)



## 2. [10 pts] Generate an ensemble of 100 classifiers for each of the four basic classifiers in Q1 and store each ensemble as a list. In order to create weak sub-classifiers within our ensembles, we underpower some hyperparameters:  
## • For the neural network, set the hidden sizes to (3, 3), max iterations to 30, and tolerance to 1e-1. 
## • For the decision tree, set max depth to 5 and max features to 5. 
## For each of these 4 ensembles, report the performance of the first classifier in the ensemble. I.e., for your ensemble of 100 decision trees, report the performance of just the first weak tree. 

## NAIVE BAYES ENSEMBLE
# Main Priors
counts = np.unique(y, return_counts=True)
model_config = {"priors":[counts[1][0]/len(y), counts[1][1]/len(y)]}
ensemble_nb = [GaussianNB(**model_config) for _ in range(100)]
acc, conf = train(X, y, model=ensemble_nb[0], k=1, shuffle=True)

## SVM ENSEMBLE
model_config = {"dual":False}
ensemble_svm = [LinearSVC(**model_config) for _ in range(100)]
acc, conf = train(X, y, model=ensemble_svm[0], k=1, shuffle=True)

## NEURAL NET ENSEMBLE
model_config = {"hidden_layer_sizes":(3,3), "max_iter":30, "tol":1e-1}
ensemble_nn = [MLPClassifier(**model_config) for _ in range(100)]
acc, conf = train(X, y, model=ensemble_nn[0], k=1, shuffle=True)

## DECISION TREE ENSEMBLE
model_config = {"max_depth":5, "max_features":5}
ensemble_dtree = [DecisionTreeClassifier(**model_config) for _ in range(100)]
acc, conf = train(X, y, model=ensemble_dtree[0], k=1, shuffle=True)



## 3. [20 pts] Write a function ensemble_fit() to receive the ensemble (i.e. one of the 4 lists from Q2.) as an input and train it on one of the subsets (i.e. bagging) of the training data. (Hint: random.sample could generate the subset of data you’ll need.) This way, each classifier will see only a different subset of the training dataset, also called as subsampling the input data for training. Use all features in these subsamples; only subsample the rows/observations. 
import random

def ensemble_fit(X, y, ensemble:list, subsample_ratio=0.2):
    indices = list(np.arange(len(X)))
    k = max(int(len(X) * subsample_ratio), 1)
    for model in ensemble:
        ## TRAIN MODEL
        idx = stratified_sample(y, indices, k)
        model.fit(X[idx], y[idx])
    return ensemble

def stratified_sample(y, indices, k):
    ## RECURSIVE SAMPLE UNTIL BOTH CLASSES ARE PRESENT
    idx = random.sample(indices, k=k)
    if np.all(y[idx] == 0) or np.all(y[idx] == 1): 
        return stratified_sample(y, indices, k)
    return idx


## 4. [20 pts] Write a function ensemble_predict() to receive the trained ensemble (i.e. one of the lists from Q3.) as input and output a prediction for a given observation. Since each sub-classifier will have its own prediction, use a voting scheme on the returned predictions. 
## (Hint: The final prediction should be the np.argmax() of the votes, not merely a “max”. Note that c.predict_proba() should have better results.) 

def ensemble_predict(X, ensemble:list):
    ## ASSUMING BINARY CLASSIFICATION
    votes = np.zeros((len(X), 2))
    for model in ensemble:
        vote = model.predict(X).astype(int)     # Binary prediction
        vote = np.stack((1-vote, vote), axis=1) # Separate columns for each class
        votes += vote                           # Add model vote to total per class
    return np.argmax(votes, axis=1)  



## 5. [20 pts] Report 10-fold CV performances of the ensembles with a subsample ratio of 0.2. Compare to a regular decision tree (same subsample ratio). Now repeat these for a subsample ratio of 0.05. 

def ensemble_train(X, y, model_class, model_config, n_estimators=100, k=10, shuffle=False, subsample_ratio=0.2, log=True):
    ## K-FOLD CROSS VALIDATION
    acc, conf = [], []
    kfold = KFold(n_splits=k, shuffle=shuffle)
    for i_train, i_test in kfold.split(X, y):
        ## TRAIN MODEL
        ensemble = [model_class(**model_config) for _ in range(n_estimators)]
        ensemble = ensemble_fit(X[i_train], y[i_train], ensemble, subsample_ratio)
        y_pred = ensemble_predict(X[i_test], ensemble)
        ## COMPUTE METRICS
        conf.append(metrics.confusion_matrix(y[i_test], y_pred))
        acc.append(metrics.accuracy_score(y[i_test], y_pred))

    ## REPORT METRICS
    conf = np.array(conf)
    if log:
        tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
        fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
        print(f"{ensemble[0].__class__.__name__} Ensemble SSR={subsample_ratio} Accuracy: {100*np.mean(acc):.4} ±{100*np.std(acc):.4}%")
        print(f"    True Positive Rate: {100*np.mean(tpr):.4}%")
        print(f"    False Positive Rate: {100*np.mean(fpr):.4}%\n")  
    return acc, conf


## NAIVE BAYES ENSEMBLE PREDICTION
counts = np.unique(y, return_counts=True)
model_config = {"priors":[counts[1][0]/len(y), counts[1][1]/len(y)]}
acc, conf = ensemble_train(X, y, GaussianNB, model_config, n_estimators=100, subsample_ratio=0.05)

## SVM ENSEMBLE PREDICTION
model_config = {"kernel":"linear", "probability":True}
acc, conf = ensemble_train(X, y, SVC, model_config, n_estimators=100, subsample_ratio=0.05)

## NEURAL NET ENSEMBLE PREDICTION
model_config = {"hidden_layer_sizes":(3,3), "max_iter":30, "tol":1e-1}
acc, conf = ensemble_train(X, y, MLPClassifier, model_config, n_estimators=100, subsample_ratio=0.05)

## DECISION TREE ENSEMBLE PREDICTION
model_config = {"max_depth":5, "max_features":5}
acc, conf = ensemble_train(X, y, DecisionTreeClassifier, model_config, n_estimators=100, subsample_ratio=0.05)



## 6. [10 pts] Report the 10-fold CV performances of the ensembles for the training subsample ratios of (0.005, 0.01, 0.03, 0.05, 0.1, 0.2). Now train regular versions of those 4 classifiers and report their performance. (Hint: pass the regular classifier to the same ensemble CV in a list of one element. This way, the same script can be used for this entire step) 

ssr = (0.005, 0.01, 0.03, 0.05, 0.1, 0.2)
acc_ensemble_nb = []
acc_ensemble_svm = []
acc_ensemble_nn = []
acc_ensemble_dtree = []

for ratio in ssr:
    ## NAIVE BAYES ENSEMBLE PREDICTION
    counts = np.unique(y, return_counts=True)
    model_config = {"priors":[counts[1][0]/len(y), counts[1][1]/len(y)]}
    acc, conf = ensemble_train(X, y, GaussianNB, model_config, n_estimators=100, subsample_ratio=ratio, log=False)
    acc_ensemble_nb.append(np.mean(acc))

    ## SVM ENSEMBLE PREDICTION
    model_config = {"kernel":"linear", "probability":True}
    acc, conf = ensemble_train(X, y, SVC, model_config, n_estimators=100, subsample_ratio=ratio, log=False)
    acc_ensemble_svm.append(np.mean(acc))

    ## NEURAL NET ENSEMBLE PREDICTION
    model_config = {"hidden_layer_sizes":(3,3), "max_iter":30, "tol":1e-1}
    acc, conf = ensemble_train(X, y, MLPClassifier, model_config, n_estimators=100, subsample_ratio=ratio, log=False)
    acc_ensemble_nn.append(np.mean(acc))

    ## DECISION TREE ENSEMBLE PREDICTION
    model_config = {"max_depth":5, "max_features":5}
    acc, conf = ensemble_train(X, y, DecisionTreeClassifier, model_config, n_estimators=100, subsample_ratio=ratio, log=False)
    acc_ensemble_dtree.append(np.mean(acc))

print(f"Naive Bayes: {', '.join([str(round(i,2)) for i in acc_ensemble_nb])}")
print(f"Linear SVC: {', '.join([str(round(i,2)) for i in acc_ensemble_svm])}")
print(f"Nueral Net: {', '.join([str(round(i,2)) for i in acc_ensemble_nn])}")
print(f"Decision Tree: {', '.join([str(round(i,2)) for i in acc_ensemble_dtree])}")


acc_nb = []
acc_svm = []
acc_nn = []
acc_dtree = []

for ratio in ssr:
    ## NAIVE BAYES ENSEMBLE PREDICTION
    counts = np.unique(y, return_counts=True)
    model_config = {"priors":[counts[1][0]/len(y), counts[1][1]/len(y)]}
    acc, conf = ensemble_train(X, y, GaussianNB, model_config, n_estimators=1, subsample_ratio=ratio, log=False)
    acc_nb.append(np.mean(acc))

    ## SVM ENSEMBLE PREDICTION
    model_config = {"kernel":"linear", "probability":True}
    acc, conf = ensemble_train(X, y, SVC, model_config, n_estimators=1, subsample_ratio=ratio, log=False)
    acc_svm.append(np.mean(acc))

    ## NEURAL NET ENSEMBLE PREDICTION
    model_config = {"hidden_layer_sizes":(3,3), "max_iter":30, "tol":1e-1}
    acc, conf = ensemble_train(X, y, MLPClassifier, model_config, n_estimators=1, subsample_ratio=ratio, log=False)
    acc_nn.append(np.mean(acc))

    ## DECISION TREE ENSEMBLE PREDICTION
    model_config = {"max_depth":5, "max_features":5}
    acc, conf = ensemble_train(X, y, DecisionTreeClassifier, model_config, n_estimators=1, subsample_ratio=ratio, log=False)
    acc_dtree.append(np.mean(acc))

print(f"Naive Bayes: {', '.join([str(round(i,2)) for i in acc_nb])}")
print(f"Linear SVC: {', '.join([str(round(i,2)) for i in acc_svm])}")
print(f"Nueral Net: {', '.join([str(round(i,2)) for i in acc_nn])}")
print(f"Decision Tree: {', '.join([str(round(i,2)) for i in acc_dtree])}")


## 7. [10 pts] For each of the 4 types of classifier, plot the performances of the ensemble at the different subsample ratios and the performances of the regular classifier at the different subsample ratios on the same plot. Thus, you should have 4 plots, one for each type of classifier. To make it graphically clear which performances are ensemble vs. regular, plotting in 2 different colors is recommended. 

def plot(ax, ssr, ensemble, regular, title):
    ax.plot(ssr, ensemble, label="Ensemble")
    ax.plot(ssr, regular, label="Single Model")
    ax.legend()
    ax.set_title(title)
    return ax


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
plot(axes[0,0], ssr, acc_ensemble_nb, acc_nb, "Naive Bayes")
plot(axes[0,1], ssr, acc_ensemble_svm, acc_svm, "Linear SVC")
plot(axes[1,0], ssr, acc_ensemble_nn, acc_nn, "Neural Net")
plot(axes[1,1], ssr, acc_ensemble_dtree, acc_dtree, "Decision Tree")

fig.suptitle("Subsample Ratio vs Accuracy per Model Ensemble")
fig.supxlabel("Subsample Ratio")
fig.supylabel("10-Fold Mean Accuracy")
