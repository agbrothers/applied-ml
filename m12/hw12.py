## DATASET EXPLORATION
import numpy as np
import pandas as pd 
import seaborn as sns; 
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
pd.options.display.max_rows = 100
plt.rcParams["figure.dpi"] = 300
sns.set(style="ticks", color_codes=True) 

## DATA PIPELINE
df = pd.read_csv("m12/creditcard.csv")
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



## CHECK LABELS
n,f = df.shape
classes = set(df["Class"])
assert len(classes) > 1

class_balance = {key: round(100*len(df[df["Class"] == key])/n, 3) for key in classes}
print("CLASS BALANCE PERCENTAGES:", class_balance)



## DROP DUPLICATES
df.drop_duplicates(inplace=True)

## MIN/MAX NORMALIZE DATA
for col in df.columns:
    if col == "Class": continue
    df[col] = df[col].astype(np.float32)
    mx = df[col].max()
    mn = df[col].min()
    df[col] = (df[col] - mn) / (mx-mn+1e-10)

## PRINT CLEAN DF
print(pd.concat((
    df.iloc[0], 
    df.dtypes, 
    df.max(), 
    df.min(), 
    df.std()
    ), keys=["Sample", "Datatypes", "Max", "Min", "Std"], axis=1)
)









from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics


def f_score(conf):
    tp = conf[1,1]
    fp = conf[1,0]
    fn = conf[0,1]
    return (2*tp) / (2*tp+fp+fn)

def train(X, y, model_class=None, model_config=None, model=None, shuffle=False, split=0.5):
    assert (model_class and model_config is not None) or model, \
        "Must provide an initialized model or a class and config"
        
    ## TRAIN MODEL
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, train_size=split, random_state=42)
    model = model or model_class(**model_config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ## COMPUTE METRICS
    fsc = f_score(metrics.confusion_matrix(y_test, y_pred))
    acc = metrics.accuracy_score(y_test, y_pred)

    print(f"{model.__class__.__name__} Accuracy: {100*acc:.4}%")
    print(f"  F-Score: {fsc:.4}%")
    return acc, fsc


X = df.drop(columns=["Class"]).values
y = df["Class"].values


## SVC
model_config = {"kernel":"linear", "probability":True}
acc, fsc = train(X, y, SVC, model_config, split=0.1, shuffle=True)
