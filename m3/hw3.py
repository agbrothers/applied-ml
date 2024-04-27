
# From the Kaggle  web site (https://www.kaggle.com/datasets)  download  the Suicide Rates 
# Overview 1985 to 2016 dataset. This  dataset  has 12 features  and 27820  data points.  In this 
# assignment  we would  like to develop  a machine learned  model to predict, given some feature 
# vectors, if the outcome would be suicide  or not, as a binary dependent  variable.  The binary 
# categories  could be {"low suicide  rate",  "high suicide  rate"}.  (Note that a different approach 
# could seek to generate  a numerical  value by solving  a regression  problem.) 
 
# A machine learning  solution  would  require  us to pre-process  the dataset  and prepare/design 
# our experimentation. 
 
# Load  the dataset  in your model development  framework (Jupyter notebook)  and examine the 
# features.  Note that  the Kaggle website  also  has  histograms  that you can inspect.  However, you 
# might want to look at the data grouped  by some other  features. For example,  what does the 
# 'number of suicides / 100k' histogram look like from country to country? 
 
# To answer the following  questions,  you have to think thoroughly,  and possibly  attempt some 
# pilot  experiments.  There is no one right or wrong answer  to some questions  below,  but you will 
# always need  to work from the data  to build  a convincing argument for your audience. 


import math
import numpy as np 
import matplotlib.pyplot as plt; plt.rcParams["figure.dpi"] = 72
import seaborn as sns; sns.set(style="ticks", color_codes=True) 
import pandas as pd 

df = pd.read_csv("m3/master.csv")


## BINARY CLASSIFIER
## SUICIDE RATE: High or Low

## DEPENDENT VARIABLE ANALYSIS
rate = df["suicides/100k pop"]
hist = rate.hist(bins=100)
hist.set_title("'suicides/100k pop' Distribution")
hist.set_xlabel("Number of Suicides per 100k Population")
hist.set_ylabel("Number of Samples")

mean = rate.mean()
median = rate.median()
mode = rate.mode()
std = rate.std()

# ## Distribution appears to be approximately exponential
# ## Thus ~ P(x) = µe^-µx
# ## We want anything over the 75th percentile to be 'high suicide rate'
# ##    P(x > k) = 1 - e^-µx = 0.75, solve for x
# p = 0.75
# percentile_k = math.log(1 - p) / -mean
# percentile_k = math.log(1 - 0.95) / -mean

## NOMINAL TO ONE-HOT
## Drop derivative features (' gdp_for_year ($) ', 'country-year', 'generation')
## Drop country as the large number of categoricals makes 
## the correlation matrix visualization hard to read. 
numeric_df = df.drop(columns=[' gdp_for_year ($) ', 'country-year', 'country', 'generation'])
numeric_df = pd.get_dummies(numeric_df, columns=['sex', 'age'])

## MOVE DEPENDENT VARIABLE TO LAST COLUMN
cols = numeric_df.columns.tolist()
cols.remove("suicides/100k pop")
cols.append("suicides/100k pop")
numeric_df = numeric_df[cols]

## CORRELATION
heatmap = sns.heatmap(numeric_df.corr(), cmap="bwr")
heatmap.set_title("Correlation Matrix")
plt.tight_layout()

## CLEANING
pct_na = (df.shape[0] - df["HDI for year"].notna().sum()) / df.shape[0]
numeric_df["HDI for year"] = numeric_df["HDI for year"].fillna(0)
numeric_df.plot(x='year', y='HDI for year', style='.')

## PLOT TIME SERIES
clean_df = df.drop(columns=[' gdp_for_year ($) ', 'country-year', 'generation','HDI for year'])
clean_df = pd.get_dummies(clean_df, columns=['sex', 'age','country'])
clean_df.groupby("year").sum().plot(y='suicides_no', fontsize=8)
clean_df.groupby("year").sum().plot(y='population', fontsize=8)
clean_df.groupby("year").sum().plot(y='suicides/100k pop', fontsize=8)
clean_df.groupby("year").sum().plot(y='gdp_per_capita ($)', fontsize=8)
clean_df.groupby("year").sum().plot(y=['age_15-24 years', 'age_25-34 years', 'age_35-54 years', 'age_5-14 years', 'age_55-74 years', 'age_75+ years'])

## COMPUTE DEPENDENT VARIABLE BINS
clean_df["high_suicide_rate"] = clean_df["suicides/100k pop"] > clean_df["suicides/100k pop"].median()

## REMOVE VARIABLES FROM WHICH THE DEPENDENT VARIABLE IS DERIVED
clean_df = clean_df.drop(columns=['suicides_no', 'suicides/100k pop'])

## REMOVE 2016 DATA
clean_df = clean_df[clean_df["year"] != 2016]

## MIN/MAX NORMALIZE DATA
for col in clean_df.columns:
    clean_df[col] = clean_df[col].astype(np.float32)
    mx = clean_df[col].max()
    mn = clean_df[col].min()
    clean_df[col] = (clean_df[col] - mn) / (mx-mn+1e-10)


## CROSS VALIDATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # f1_score can be used too
from sklearn.model_selection import KFold, train_test_split    

# We will reuse the classifier function below
def rf_train_test(_X_tr, _X_ts, _y_tr, _y_ts):
    # Create a new random forest classifier, with working 4 parallel cores
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0, n_jobs=4)
    # Train on training data
    rf.fit(_X_tr, _y_tr)
    # Test on training data
    y_pred = rf.predict(_X_ts)
    # Return more proper evaluation metric
    # return f1_score(_y_ts, y_pred, pos_label='recurrence-events', zero_division=0)
    # Return accuracy
    return accuracy_score(_y_ts, y_pred)

# 10-fold cross validation
# Prepare the input X matrix and target y vector
X = clean_df.loc[:, clean_df.columns != 'high_suicide_rate'].values
y = clean_df.loc[:, clean_df.columns == 'high_suicide_rate'].values.ravel()

history = []
kfold = KFold(n_splits=10,shuffle=False)
for i_train, i_test in kfold.split(X, y):
    acc = rf_train_test(X[i_train], X[i_test], y[i_train], y[i_test])
    history.append(acc)

print(f'10-fold cross validation accuracy is {np.mean(history):.3f} {chr(177)} {np.std(history):.4f}')
