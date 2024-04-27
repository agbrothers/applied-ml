## DATASET EXPLORATION
import pandas as pd 
import seaborn as sns; 
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.figsize"] = (8,6)
sns.set(style="ticks", color_codes=True) 

# examine a single sample
df_train = pd.read_csv("m5/train.csv")
sample = df_train.iloc[0]
dtypes = df_train.dtypes
nans = 100 * df_train.isna().sum() / len(df_train)
categories = df_train.nunique()
print(pd.concat(
    (sample, dtypes, nans, categories), 
    keys=["Sample", "Datatypes", "NaN %", "Num Categories"], 
    axis=1)
)

# examine the categorical features
print("\nTicket values: ")
print(set(df_train["Ticket"].values[:20]))
print("\nCabin values: ")
print(set(df_train["Cabin"].values))
print("\nEmbarked values: ")
print(set(df_train["Embarked"].values))

## DROP IRRELEVANT OR ERRONEOUS FEATURES
features_to_drop = ["PassengerId","Name","Ticket","Cabin"]
df = df_train.drop(columns=features_to_drop)

## RE-TYPE CATEGORICAL VARIABLES
numeric = ["Age","Fare","SibSp","Parch"]
categorical = ["Survived","Pclass","Sex","Embarked"]
for feat in categorical: 
    df[feat] = df[feat].astype('category')

## CONVERT CATEGORICAL VARIABLES TO NUMERIC REPRESENTATIONS
hist_df = df.copy()
hist_df[categorical] = hist_df[categorical].apply(lambda x: x.cat.codes)
print(pd.concat((hist_df.iloc[0], hist_df.dtypes), keys=["Sample", "Datatypes"], axis=1))

# examine how the features are distributed
plt.clf()
df.hist(bins=50, figsize=(8,8), layout=(4,2))
plt.tight_layout()


## EXAMINE CORRELATIONS BETWEEN FEATURES
df = pd.get_dummies(df)

plt.clf()
plt.rcParams["figure.figsize"] = (4,4)
heatmap = sns.heatmap(df_train.corr(), cmap="bwr")
heatmap.set_title("Correlation Matrix")
plt.tight_layout()


## NEAREST NEIGHBOR IMPUTATION
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

plt.clf()
df.hist(column=["Age","Embarked_C", "Embarked_Q", "Embarked_S"], bins=50, figsize=(8,4), layout=(2,2))
plt.tight_layout()

## TRAIN MODEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold    
from sklearn import metrics
import numpy as np 

def train(model_class, model_config, X, y, k=10, shuffle=False):
    ## K-FOLD CROSS VALIDATION
    acc, conf = [], []
    kfold = KFold(n_splits=k, shuffle=shuffle)
    for i_train, i_test in kfold.split(X, y):
        
        ## TRAIN MODEL
        model = model_class(**model_config)
        model.fit(X[i_train], y[i_train])
        y_pred = model.predict(X[i_test])

        ## COMPUTE METRICS
        conf.append(metrics.confusion_matrix(y[i_test], y_pred))
        acc.append(metrics.accuracy_score(y[i_test], y_pred))

    return acc, conf

X = df.drop(columns=["Survived_0","Survived_1"]).values
y = df["Survived_1"].values
model_config = {"n_estimators":10, "max_features":8}
acc, conf = train(RandomForestClassifier, model_config, X, y, k=10, shuffle=True)


## ANALYZE RESULTS
print(f"Accuracy per fold: {'%, '.join(np.round(100*np.array(acc),2).astype(str))}")
print(f"10-Fold Mean Accuracy: {100*np.mean(acc):.4}%")

conf = np.array(conf)
tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
print(f"10-Fold Mean True Positive Rate: {100*np.mean(tpr):.4}%")
print(f"10-Fold Mean False Positive Rate: {100*np.mean(fpr):.4}%")


## RETRAIN FINAL MODEL ON FULL DATASET
model = RandomForestClassifier(**model_config)
model.fit(X, y)
y_pred = model.predict(X)

final_acc = metrics.accuracy_score(y, y_pred)
final_conf = metrics.confusion_matrix(y, y_pred)
final_tpr = final_conf[0,0] / (final_conf[0,0] + final_conf[1,0] + 1e-10)
final_fpr = final_conf[0,1] / (final_conf[0,0] + final_conf[0,1] + 1e-10)
print(f"Full Training Set Accuracy: {100*final_acc:.4}%")
print(f"Full Training Set True Positive Rate: {100*final_tpr:.4}%")
print(f"Full Training Set False Positive Rate: {100*final_fpr:.4}%")




## PRE-PROCESS TEST DATASET
df_test = pd.read_csv("m5/test.csv")                    
sample = df_test.iloc[0]
dtypes = df_test.dtypes
nans = 100 * df_test.isna().sum() / len(df_test)
categories = df_test.nunique()
print(pd.concat(
    (sample, dtypes, nans, categories), 
    keys=["Sample", "Datatypes", "NaN %", "Num Categories"], 
    axis=1)
)

## DROP FEATURES
df_passenger_id = pd.DataFrame(df_test["PassengerId"])
df_test = df_test.drop(columns=features_to_drop)                                                

## ONE-HOT ENCODING
for feat in categorical[1:]:                            
    df_test[feat] = df_test[feat].astype('category')                                       
df_test = pd.get_dummies(df_test)                       

## EXAMINE AGE BEFORE IMPUTING
plt.clf()
df_test.hist(column=["Age"], bins=50, figsize=(8,4), layout=(2,2))
plt.tight_layout()

## FIT A NEW IMPUTER ON BOTH THE TRAIN AND TEST SET
df_combined = pd.concat((df, df_test)).drop(columns=["Survived_0","Survived_1"])
test_imputer = KNNImputer(n_neighbors=3)
test_imputer.fit_transform(df_combined)

## APPLY IMPUTER TO THE TEST SET
df_test = pd.DataFrame(                                 
    test_imputer.transform(df_test),                    
    columns=df_test.columns                             
)

## EXAMINE AGE BEFORE IMPUTING
plt.clf()
df_test.hist(column=["Age"], bins=50, figsize=(8,4), layout=(2,2))
plt.tight_layout()

## EXAMINE FEATURE DISTRIBUTIONS AFTER PRE-PROCESSING
plt.clf()
df_test.hist(bins=50, figsize=(8,8), layout=(4,2))
plt.tight_layout()

## PRODUCE TEST SET PREDICTIONS
X_test = df_test.values
y_test_pred = model.predict(X_test).astype(int)

def save_preds(filepath, y_pred, df): 

    import csv 
    with open(filepath, 'w') as fout: 
        writer = csv.writer(fout, delimiter=',', lineterminator='\n') 
        writer.writerow(['PassengerId', 'Survived']) 
        for yid, ypred in zip(df['PassengerId'], y_pred): 
            writer.writerow([yid, ypred]) 
       
save_preds('predictions_brothers.csv', y_pred, df_passenger_id) 

