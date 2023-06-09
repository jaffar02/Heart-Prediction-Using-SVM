import pandas as pd                                  # pandas is used to load and manipulate data and for One-Hot Encoding
import numpy as np                                   # numpy is used to calculate the mean and standard deviation

import matplotlib.pyplot as plt                      # matplotlib is for drawing graphs
import matplotlib.colors as colors


import pickle
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


from sklearn.model_selection import train_test_split                # split  data into training and testing sets
from sklearn.model_selection import GridSearchCV                    # this will do cross validation
from sklearn.preprocessing import scale                             # scale and center data
from sklearn.svm import SVC                                         # this will make a support vector machine for classificaiton
from sklearn.metrics import confusion_matrix, classification_report # this creates a confusion matrix
#from sklearn.metrics import plot_confusion_matrix                  # draws a confusion matrix
from sklearn.decomposition import PCA                               # to perform PCA to plot the data

df = pd.read_csv(open('heart-disease-prediction\Misc\processed.cleveland.data', 'rb'))
df.head()

df.columns = ['age',
              'sex',
              'cp',
              'restbp',
              'chol',
              'fbs',
              'restecg',
              'thalach',
              'exang',
              'oldpeak',
              'slope',
              'ca',
              'thal',
              'hd']


#processing the dataset by removing null values
df_no_missing = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]
X = df_no_missing.drop('hd', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1].copy()
y = df_no_missing['hd'].copy()


#for making our label column having only 1 or 0
y_not_zero_idx = y > 0
y[y_not_zero_idx] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

clf_svm = SVC(random_state=42, C=1, gamma=0.01).fit(X_train_scaled, y_train)
y_pred = clf_svm.predict(X_test_scaled)
print(y_pred)