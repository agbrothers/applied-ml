# Assignment 11 
# Applied Machine Learning 
# 1. [50 pts] In this assignment, we will use a priori analysis to find phrases, or interesting word 
# patterns, in a novel. 
 
# Note that you are free to use any a priori analytics and algorithm library in this assignment. 
 
# Use the nltk library corpus gutenberg API and load the novel 'carroll-alice.txt', which is 
# Alice in Wonderland by Lewis Carroll (although his real name was Charles Dodgson). 
# There are 1703 sentences in the novel—which can be represented as 1703 transactions. 
# Use any means you like to parse/extract words and save in a .csv format to be read by 
# Weka framework, similar to the a priori Analysis module. (Hint: Feel free to use mlxtend 
# library instead of Weka.) 
 
# Hint: Removing stop words and symbols using regular expressions can be helpful: 
# from nltk.corpus import gutenberg, stopwords 
# Stop_words = stopwords.words('english') 
# Sentences = gutenberg.sents('carroll-alice.txt') 
# TermsSentences = [] 
# for terms in Sentences: 
#   terms = [w for w in terms if w not in Stop_words] 
#   terms = [w for w in terms if re.search(r'^[a-zA-Z]{2}', w) is not None] 
 
# If you chose to Weka, use FPGrowth and start with default parameters. Reduce 
# lowerBoundMinSupport to reach to a sweet point for the support and avoid exploding the 
# number of rules generated. 
 
# Report interesting patterns. 
 
# (Example: Some of the frequently occurring phrases are “Mock Turtle”, “White Rabbit”, etc.) 

"""
import re
import nltk
from nltk.corpus import gutenberg, stopwords
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

## DOWNLOAD NLTK DATA
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')

## LOAD STOPWORDS + SENTENCES
stopwords = set(stopwords.words('english'))
sentences = gutenberg.sents('carroll-alice.txt')

## REMOVE STOPWORDS AND SYMBOLS VIA REGEX
processed_sentences = []
for terms in sentences:
    terms = [w.lower() for w in terms if w.lower() not in stopwords]
    terms = [re.sub(r'[^a-zA-Z]', '', w) for w in terms if re.search(r'^[a-zA-Z]{2}', w)]
    processed_sentences.append(terms)

## ENCODE SENTENCES LIKE TRANSACTIONS
encoder = TransactionEncoder()
onehot = encoder.fit(processed_sentences).transform(processed_sentences)
df = pd.DataFrame(onehot, columns=encoder.columns_)

## ENCODE SENTENCES LIKE TRANSACTIONS
encoder = TransactionEncoder()
onehot = encoder.fit(processed_sentences).transform(processed_sentences)
df = pd.DataFrame(onehot, columns=encoder.columns_)

## APPLY APRIORI ALGORITHM TO GET FREQUENT ITEMS
freq_itemsets = apriori(df, min_support=0.01, use_colnames=True)
freq_itemsets.sort_values(by='support', ascending=False).head(30)
"""





# 2. [50 pts] In the lecture module, the class NeuralNetMLP implements a neural network with a 
# single hidden layer. Make the necessary modifications to upgrade it to a 2-hidden layer 
# neural network. Run it on the MNIST dataset and report its performance. 
 
# (Hint: Raschka, Chapter 12) 

# curl 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' > train-images-idx3-ubyte.gz
# curl 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' > train-labels-idx1-ubyte.gz
# curl 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' > t10k-images-idx3-ubyte.gz
# curl 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' > t10k-labels-idx1-ubyte.gz
 
## LOAD DATASET
import numpy as np

def load_mnist(path, kind='train'):
    from numpy import fromfile, uint8
    import os
    import struct
    
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = fromfile(lbpath, dtype=uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
            images = fromfile(imgpath, dtype=uint8).reshape(len(labels), 784)
            images = ((images / 255.) - .5) * 2
    return images, labels

X_train_mnist, y_train_mnist = load_mnist('m11/mnist/', kind='train')
print(f'Rows= {X_train_mnist.shape[0]}, columns= {X_train_mnist.shape[1]}')

X_test_mnist, y_test_mnist = load_mnist('m11/mnist/', kind='t10k')
print(f'Rows= {X_test_mnist.shape[0]}, columns= {X_test_mnist.shape[1]}')



## MLP CLASS FROM LECTURE 
## MODIFIED WITH A SECOND HIDDEN LAYER
import numpy as np

class NeuralNetMLP(object):

    def __init__(self, n_hidden=30, epochs=100, eta=0.001, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)  # used to randomize weights
        self.n_hidden = n_hidden  # size of the hidden layer
        self.epochs = epochs  # number of iterations
        self.eta = eta  # learning rate
        self.minibatch_size = minibatch_size  # size of training batch - 1 would not work
        self.w_out, self.w_h1, self.w_h2 = None, None, None
    
    @staticmethod
    def onehot(_y, _n_classes):  # one hot encode the input class y
        onehot = np.zeros((_n_classes, _y.shape[0]))
        for idx, val in enumerate(_y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T
    
    @staticmethod
    def sigmoid(_z):  # Eq 1
        return 1.0 / (1.0 + np.exp(-np.clip(_z, -250, 250)))

    def _forward(self, _X):  # Eq 2
        z_h1 = np.dot(_X, self.w_h1)
        a_h1 = self.sigmoid(z_h1)
        z_h2 = np.dot(a_h1, self.w_h2)
        a_h2 = self.sigmoid(z_h2)
        z_out = np.dot(a_h2, self.w_out)
        a_out = self.sigmoid(z_out)
        return a_h1, a_h2, a_out, z_out

    @staticmethod
    def compute_cost(y_enc, output):  # Eq 4
        term1 = -y_enc * (np.log(output))
        term2 = (1.0-y_enc) * np.log(1.0-output)
        cost = np.sum(term1 - term2)
        return cost

    def predict(self, _X):
        a_h1, a_h2, a_out, z_out = self._forward(_X)
        ypred = np.argmax(z_out, axis=1)
        return ypred

    def fit(self, _X_train, _y_train, _X_valid, _y_valid):
        import sys
        n_output = np.unique(_y_train).shape[0]  # number of class labels
        n_features = _X_train.shape[1]
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))
        self.w_h2 = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_hidden))
        self.w_h1 = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        y_train_enc = self.onehot(_y_train, n_output)  # one-hot encode original y
        for ei in range(self.epochs):  # Ideally must shuffle at every epoch
            indices = np.arange(_X_train.shape[0])
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                
                a_h1, a_h2, a_out, z_out = self._forward(_X_train[batch_idx])  # neural network model
                
                sigmoid_derivative_h2 = a_h2 * (1.0-a_h2)  # Eq 3
                sigmoid_derivative_h1 = a_h1 * (1.0-a_h1)  # Eq 3
                
                delta_out = a_out - y_train_enc[batch_idx]  # Eq 5
                delta_h2 = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h2)  # Eq 6
                delta_h1 = (np.dot(delta_h2, self.w_h2.T) * sigmoid_derivative_h1)  # Eq 6
                
                grad_w_out = np.dot(a_h2.T, delta_out)  # Eq 7
                grad_w_h2 = np.dot(a_h1.T, delta_h2)  # Eq 8
                grad_w_h1 = np.dot(_X_train[batch_idx].T, delta_h1)  # Eq 8
                
                self.w_out -= self.eta*grad_w_out  # Eq 9
                self.w_h2 -= self.eta*grad_w_h2  # Eq 9
                self.w_h1 -= self.eta*grad_w_h1  # Eq 9

            # Evaluation after each epoch during training
            a_h1, a_h2, a_out, z_out = self._forward(_X_train)
            cost = self.compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(_X_train)  # monitoring training progress through reclassification
            y_valid_pred = self.predict(_X_valid)  # monitoring training progress through validation
            train_acc = ((np.sum(_y_train == y_train_pred)).astype(float) / _X_train.shape[0])
            valid_acc = ((np.sum(_y_valid == y_valid_pred)).astype(float) / _X_valid.shape[0])
            sys.stderr.write('\r%d/%d | Cost: %.2f ' '| Train/Valid Acc.: %.2f%%/%.2f%% '%
                (ei+1, self.epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()
        return self


## TRAIN THE NN
from sklearn.metrics import confusion_matrix

def get_acc(_y_test, _y_pred):
    return (np.sum(_y_test == _y_pred)).astype(float) / _y_test.shape[0]

nn = NeuralNetMLP(n_hidden=20, epochs=300, eta=0.0005, minibatch_size=100, seed=1)
nn.fit(X_train_mnist[:55000], y_train_mnist[:55000], X_train_mnist[55000:], y_train_mnist[55000:]) ;

y_pred = nn.predict(X_test_mnist)

print(f'Accuracy= {get_acc(y_test_mnist, y_pred)*100:.2f}%')
print(confusion_matrix(y_test_mnist, y_pred))


