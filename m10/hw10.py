## Assignment 10 
## Applied Machine Learning 
## In this assignment, we will develop a ML model for cybersecurity intrusion detection. Please visit the website: https://www.unb.ca/cic/datasets/ids-2017.html and look around to see the problem space and the evaluation datasets to be used for ML model development. 

## This dataset was collected by cyber experts during experimentation that was carried out over the course of 5 days. The description of the experiments also informs the experimental ground truth. 
## (Suggested: GeneratedLabelledFlows.zip; note that it is already pre-processed by someone) 
 
## 1. [10 pts] Download the labeled dataset (if you like, use a dummy email address for registration). There must be 8 data files, each representing a particular cyber-attack type, its day, and its collected packet capture (“PCAP”) data. 




## 2. [10 pts] Pick one of the data files, call it Dataset 1, and examine its features. Make sure it has more than one class value for its label. 

import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 10
dataset_1 = "./m10/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df1 = pd.read_csv(dataset_1)

## CHECK LABELS
n,f = df1.shape
classes = set(df1[" Label"])
assert len(classes) > 1

## CHECK FEATURES
sample = df1.iloc[0]
nans = 100 * df1.isna().sum() / len(df1)
print(pd.concat(
    (sample, df1.dtypes, nans, df1.nunique(), df1.min(), df1.max()), 
    keys=["Sample", "Datatype", "NaN %", "Unique", "Min", "Max"], 
    axis=1).drop("Flow ID")
)
## REPLACE INF WITH 
import numpy as np
df1["Flow Bytes/s"][df1["Flow Bytes/s"] == np.inf] = np.nan
df1 = df1.dropna()

## 3. [10 pts] For your Dataset 1, pick a machine learning methodology and justify your choice. 




## 4. [10 pts] Process the class feature/category as binary classes for supervised learning, assign BENIGN to value 0 and the rest to value 1. Check its balance for the Dataset 1. 

## CHECK LABELS
class_balance = {key: round(100*len(df1[df1[" Label"] == key])/n, 3) for key in classes}
print("CLASS BALANCE PERCENTAGES:", class_balance)
df1[" Label"][df1[" Label"] != "BENIGN"] = 1
df1[" Label"][df1[" Label"] == "BENIGN"] = 0
df1[" Label"] = df1[" Label"].astype(int)


## 5. [10 pts] Explore Dataset 1 features with respect to the class. (Hint: features Source Port and Destination Port are very useful; research and find out important networking port numbers and one-hot-encode them. Unimportant port numbers or source port numbers can be assigned to a feature called 'other ports'.) 

## PORT CATEGORIES
## 0 to 1023      -> System services/network protocol ports
## 1024 to 49151  -> Registered ports
## 49152 to 65535 -> Dynamic/Private ports

## RELEVANT PORTS FOR DDoS ATTACKS:
## [source] https://blog.netwrix.com/2022/08/04/open-port-vulnerabilities-list/ 
ports = [
    20, 21, ## FTP: File Transfer Protocol ports that let users send and receive files from servers.
    22,     ## SSH: A TCP port for ensuring secure access to servers. Hackers can exploit port 22 by using leaked SSH keys or brute-forcing credentials.
    25,     ## SMTP: DDoS attacks targeting email servers might focus on SMTP ports to flood the server with spam emails or connection attempts.
    53,     ## DNS: DNS amplification attacks exploit misconfigured DNS servers to generate large volumes of traffic directed towards the victim's network.
    80, 443, 8080, 8443, ## HTTP/HTTPS: DDoS attacks targeting web servers often focus on these ports to flood the server with HTTP/HTTPS requests, exhausting its resources and causing denial of service.
    123,    ## NTP: Network Time Protocol (NTP) reflection attacks abuse NTP servers to amplify attack traffic towards the victim, overwhelming their resources.
    161,    ## SNMP: Simple Network Management Protocol (SNMP) reflection attacks abuse SNMP-enabled devices to amplify attack traffic towards the victim.
    1900,   ## SSDP: Simple Service Discovery Protocol (SSDP) reflection attacks exploit vulnerable SSDP-enabled devices to amplify DDoS traffic.
    5060,   ## VoIP/SIP: DDoS attacks targeting Voice over IP (VoIP) services often exploit SIP (Session Initiation Protocol) to flood the service with SIP requests, causing disruption.
    "system",
    "registered",
    "dynamic",
]
def encode_port(port):
    if port in ports:
        return ports.index(port)
    elif port >= 0 and port < 1024:
        return ports.index("system")
    elif port >= 1024 and port < 49152:
        return ports.index("registered")
    elif port >= 49152:
        return ports.index("dynamic")

df1[" Source Port"] = df1[" Source Port"].apply(encode_port)
df1[" Destination Port"] = df1[" Destination Port"].apply(encode_port)
## WILL REPLACE NUMERIC VALUE WITH ONE-HOT LATER


## 6. [10 pts] Display some histograms and anything you deem fit to pick independent Dataset 1 features. (Hint: source/destination bytes, packets, ports and the duration features.) 
import matplotlib.pyplot as plt

def plot_hist(df, col):
    df[col].hist(bins=100)
    plt.title(col)
    plt.xlabel(f"{col} values")
    plt.ylabel("Occurances")
    plt.tight_layout()

# plot_hist(df1, " Source Port")
# plot_hist(df1, " Destination Port")
# plot_hist(df1, " Flow Duration")
# plot_hist(df1, " Total Fwd Packets")
# plot_hist(df1, " Total Backward Packets")

def remove_outliers(df, col):
    return df[df[col].abs() < 3*df[col].std()]

df1 = remove_outliers(df1, "Total Length of Fwd Packets")


## 7. [10 pts] Attempt a few classifier models and report their 10-fold CV performances. 

## ONE-HOT ENCODE PORTS AND DROP IPS
df1 = pd.get_dummies(df1, columns=[' Source Port', ' Destination Port'])
df1 = df1.drop(columns=["Flow ID", " Source IP", " Destination IP", " Timestamp"])

## FINALIZE THE TRAINING SET
X = df1.drop(columns=[" Label"]).values
y = df1[" Label"].values

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

# ## TRAIN RANDOM FOREST MODEL
# model_config = {"n_estimators":10, "max_features":8}
# acc, conf = train(RandomForestClassifier, model_config, X, y, k=10, shuffle=True)

# ## ANALYZE RESULTS
# print(f"RANDOM FOREST: {dataset_1}")
# print(f"Accuracy per fold: {'%, '.join(np.round(100*np.array(acc),2).astype(str))}")
# print(f"10-Fold Mean Accuracy: {100*np.mean(acc):.4}%")

# conf = np.array(conf)
# tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
# fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
# print(f"10-Fold Mean True Positive Rate: {100*np.mean(tpr):.4}%")
# print(f"10-Fold Mean False Positive Rate: {100*np.mean(fpr):.4}%")

## 8. [10 pts] Convert your code to be used for the remaining 7 datasets, i.e., Datasets 2-8. 

def clean_and_train(path):
    ## LOAD DATA
    df = pd.read_csv(path)

    ## CHECK LABELS
    n,f = df.shape
    classes = set(df[" Label"])

    ## DROP INVALID VALUES (inf)
    df["Flow Bytes/s"][df["Flow Bytes/s"] == np.inf] = np.nan
    df = df.dropna()    

    ## CHECK CLASS BALANCE
    class_balance = {key: round(100*len(df[df[" Label"] == key])/n, 3) for key in classes}
    print("CLASS BALANCE PERCENTAGES:", class_balance)
    df[" Label"][df[" Label"] != "BENIGN"] = 1
    df[" Label"][df[" Label"] == "BENIGN"] = 0
    df[" Label"] = df[" Label"].astype(int)
    
    ## ENCODE PORTS
    df[" Source Port"] = df[" Source Port"].apply(encode_port)
    df[" Destination Port"] = df[" Destination Port"].apply(encode_port)
    df = pd.get_dummies(df, columns=[' Source Port', ' Destination Port'])

    ## REMOVE OUTLIERS
    df = remove_outliers(df, " Min Packet Length")

    ## REMOVE IRRELEVANT FEATURES
    df = df.drop(columns=["Flow ID", " Source IP", " Destination IP", " Timestamp"])

    ## EXTRACT ARRAYS
    X = df1.drop(columns=[" Label"]).values
    y = df1[" Label"].values

    ## TRAIN RANDOM FOREST MODEL
    model_config = {"n_estimators":10, "max_features":8}
    acc, conf = train(RandomForestClassifier, model_config, X, y, k=10, shuffle=True)

    ## ANALYZE RESULTS
    print(f"RANDOM FOREST: {path}")
    print(f"Accuracy per fold: {'%, '.join(np.round(100*np.array(acc),2).astype(str))}")
    print(f"10-Fold Mean Accuracy: {100*np.mean(acc):.4}%")

    conf = np.array(conf)
    tpr = conf[:, 0,0] / (conf[:, 0,0] + conf[:, 1,0] + 1e-10)
    fpr = conf[:, 0,1] / (conf[:, 0,0] + conf[:, 0,1] + 1e-10)
    print(f"10-Fold Mean True Positive Rate: {100*np.mean(tpr):.4}%")
    print(f"10-Fold Mean False Positive Rate: {100*np.mean(fpr):.4}%")
    return 


## 9. [10 pts] Pick a classifier algorithm and report its evaluation for the remaining 7 datasets. Note that one dataset has a single class, which might need an unsupervised learning.  
pd.read_csv("./m10/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding="latin1")
clean_and_train("./m10/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")


## 10. [10 pts] Briefly write up your thoughts about developing a machine learning model where you are not a subject matter expert, such as, developing a cybersecurity intrusion detection pipeline as in this assignment. 



