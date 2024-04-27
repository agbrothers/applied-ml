# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 72
import pandas as pd
import seaborn as sns; 
sns.set(style="ticks", color_codes=True) 

def mean(x):
    return sum(x) / len(x)

def std(x):
    m = mean(x)
    return (sum((i-m)**2 for i in x)/len(x))**0.5

def cov(x, y):
    N = len(x)
    x_ = mean(x)
    y_ = mean(x)
    return (sum((x[i]-x_)*(y[i]-y_) for i in range(N))/(N))

def corr(x,y):
    return cov(x,y)/(std(x)*std(y))

def plot_corr(df):
    corr_matrix = []
    for f_i in df.columns:
        corr_row = []
        for f_j in df.columns:
            corr_row.append(corr(df[f_i].values,df[f_j].values))
        corr_matrix.append(corr_row)


    corr_df = pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns) 
    assert all(corr_df - df.corr() < 1e-10)
    print(corr_df)
    sns.heatmap(corr_df, annot=True)

# Locate and load the data file
df = pd.read_csv('m2/Admission_Predict.csv')
plot_corr(df)
