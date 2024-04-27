## PREPROCESS DATASET

import numpy as np 
import pandas as pd 

df = pd.read_csv("m3/master.csv")

## DROP POOR FEATURES AND GET ONE-HOT ENCODINGS
clean_df = df.drop(columns=[' gdp_for_year ($) ', 'country-year','HDI for year', 'country'])
clean_df = pd.get_dummies(clean_df, columns=['sex', 'age', 'generation'])

## REMOVE VARIABLES FROM WHICH THE DEPENDENT VARIABLE IS DERIVED
clean_df = clean_df.drop(columns=['suicides_no'])

## REMOVE 2016 DATA
clean_df = clean_df[clean_df["year"] != 2016]

## MOVE DEPENDENT VARIABLE TO LAST COLUMN
cols = clean_df.columns.tolist()
cols.remove("suicides/100k pop")
cols.append("suicides/100k pop")
clean_df = clean_df[cols]

## MIN/MAX NORMALIZE DATA
clean_df = clean_df.astype(np.float32)
mn = clean_df.min().values
mx = clean_df.max().values
norm = lambda x, mn, mx: (x - mn) / (mx-mn+1e-10)
norm_df = norm(clean_df, mn, mx)

# Prepare the input X matrix and target y vector
X = norm_df.loc[:, norm_df.columns != 'suicides/100k pop'].values
y = norm_df.loc[:, norm_df.columns == 'suicides/100k pop'].values.ravel()



#######################
## ASSIGNMENT 7 WORK ##
#######################

## PROBLEM 1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
model_config = {"fit_intercept":True}
model = LinearRegression(**model_config)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred):.4f}")
print(f"Number of coefficients: {model.n_features_in_+1}")



## PROBLEM 2
## LOOKUP GROUND TRUTH SAMPLES WITH THE SPECIFIED FEATURES
truth = clean_df[clean_df["sex_male"] == 1.]
truth = truth[truth["age_15-24 years"] == 1.]
truth = truth[truth["generation_Generation X"] == 1.]
truth = np.stack(truth.values)[..., :-1]
norm_truth = norm(truth, mn[:-1], mx[:-1])

## PRODUCE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
y_pred = model.predict(norm_truth)
y_true = np.mean(y_pred)

## GENERATE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
sample = {key:0. for key in norm_df.keys() if key != "suicides/100k pop"}
sample["age_15-24 years"] = 1.
sample["sex_male"] = 1.
sample["generation_Generation X"] = 1.

samples = []
for _ in range(100):
    sample.update({
        "year": np.random.randint(mn[0],mx[0]),
        "population": np.random.uniform(mn[1],mx[1]),
        "gdp_per_capita ($)": np.random.uniform(mn[2],mx[2]),
    })
    x = np.array(list(sample.values()))
    x_norm = norm(x, mn[:-1], mx[:-1])
    samples.append(x_norm)
samples = np.stack(samples)

## PRODUCE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
y_pred = model.predict(samples)

mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.4f}")




## PROBLEM 3
## DROP POOR FEATURES
clean_df = df.drop(columns=[' gdp_for_year ($) ', 'country-year','HDI for year', 'country'])

## CONVERT CATEGORICAL VARIABLES TO NUMERIC
categorical = ["sex","age","generation"]
mapping = {}
mapping["sex"] = {
    "male": 0.,
    "female": 1.,
}
mapping["age"] = {
    "75+ years": 0.,
    "55-74 years": 1.,
    "35-54 years": 2.,
    "25-34 years": 3.,
    "15-24 years": 4.,
    "5-14 years": 5.,
}
mapping["generation"] = {
    "G.I. Generation": 0.,   # GI Generation – 1901-1927.
    "Silent": 1.,            # Silent Generation – 1928-1945.
    "Boomers": 2.,           # Baby Boomers – 1946-1964.
    "Generation X": 3.,      # Generation X – 1965 - 1980.
    "Millenials": 4.,        # Millennials – 1981-1996.
    "Generation Z": 5.,      # Generation Z – 1997-2012.
    "Generation Alpha": 6.,  # Generation Alpha – 2013 - present.
}

for feat in categorical: 
    clean_df[feat] = clean_df[feat].apply(lambda x: mapping[feat][x])

## REMOVE VARIABLES FROM WHICH THE DEPENDENT VARIABLE IS DERIVED
clean_df = clean_df.drop(columns=['suicides_no'])

## REMOVE 2016 DATA
clean_df = clean_df[clean_df["year"] != 2016]

## MOVE DEPENDENT VARIABLE TO LAST COLUMN
cols = clean_df.columns.tolist()
cols.remove("suicides/100k pop")
cols.append("suicides/100k pop")
clean_df = clean_df[cols]

## MIN/MAX NORMALIZE DATA
clean_df = clean_df.astype(np.float32)
mn = clean_df.min().values
mx = clean_df.max().values
norm = lambda x, mn, mx: (x - mn) / (mx-mn+1e-10)
unnorm = lambda x, mn, mx: (x * (mx-mn+1e-10)) + mn
norm_df = norm(clean_df, mn, mx)

# Prepare the input X matrix and target y vector
X = norm_df.loc[:, norm_df.columns != 'suicides/100k pop'].values
y = norm_df.loc[:, norm_df.columns == 'suicides/100k pop'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
model_config = {"fit_intercept":True}
model = LinearRegression(**model_config)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred):.4f}")
print(f"Number of coefficients: {model.n_features_in_+1}")


## PROBLEM 4
## LOOKUP GROUND TRUTH SAMPLES WITH THE SPECIFIED FEATURES
truth = clean_df[clean_df["sex"] == mapping["sex"]["male"]]
truth = truth[truth["age"] == mapping["age"]["15-24 years"]]
truth = truth[truth["generation"] == mapping["generation"]["Generation X"]]
truth = np.stack(truth.values)[..., :-1]
norm_truth = norm(truth, mn[:-1], mx[:-1])

## PRODUCE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
y_pred = model.predict(norm_truth)
y_true = np.mean(y_pred)

## GENERATE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
sample = {key:0. for key in norm_df.keys() if key != "suicides/100k pop"}
sample["sex"] = mapping["sex"]["male"]
sample["age"] = mapping["age"]["15-24 years"]
sample["generation"] = mapping["generation"]["Generation X"]

samples = []
for _ in range(100):
    sample.update({
        "year": np.random.randint(mn[0],mx[0]),
        "population": np.random.uniform(mn[1],mx[1]),
        "gdp_per_capita ($)": np.random.uniform(mn[2],mx[2]),
    })
    x = np.array(list(sample.values()))
    x_norm = norm(x, mn[:-1], mx[:-1])
    samples.append(x_norm)
samples = np.stack(samples)

## PRODUCE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
y_pred = model.predict(samples)

mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae:.4f}")




## PROBLEM 6
## GENERATE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
sample = {key:0. for key in norm_df.keys() if key != "suicides/100k pop"}
sample["sex"] = mapping["sex"]["male"]
sample["age"] = mapping["age"]["25-34 years"]
sample["generation"] = mapping["generation"]["Generation Alpha"]

samples = []
for _ in range(100):
    sample.update({
        "year": np.random.randint(mn[0],mx[0]),
        "population": np.random.uniform(mn[1],mx[1]),
        "gdp_per_capita ($)": np.random.uniform(mn[2],mx[2]),
    })
    x = np.array(list(sample.values()))
    x_norm = norm(x, mn[:-1], mx[:-1])
    samples.append(x_norm)
samples = np.stack(samples)

## PRODUCE RANDOM SAMPLES WITH THE SPECIFIED FEATURES
y_pred = model.predict(samples)
y_unnorm = unnorm(np.mean(y_pred), mn[-1], mx[-1])
print(f"Predicted suicides/100k pop: {y_unnorm:.4f}")






























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

acc, conf = train(LinearRegression, model_config, X, y, k=10, shuffle=True)

## ANALYZE RESULTS
print(f"Accuracy per fold: {'%, '.join(np.round(100*np.array(acc),2).astype(str))}")
print(f"10-Fold Mean Accuracy: {100*np.mean(acc):.4}%")

conf = np.array(conf)
tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
print(f"10-Fold Mean True Positive Rate: {100*np.mean(tpr):.4}%")
print(f"10-Fold Mean False Positive Rate: {100*np.mean(fpr):.4}%")





