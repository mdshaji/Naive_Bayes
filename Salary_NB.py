# Importing Required Packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
Train = pd.read_csv("D:/Module 19/SalaryData_Train.csv")
Test = pd.read_csv("D:/Module 19/SalaryData_Test.csv")
string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])
        
##Capturing the column names which can help in futher process
colnames = Train.columns
colnames
len(colnames)

x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(x_train, y_train)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(x_test)
accuracy_test_m = np.mean(test_pred_m == y_test)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, y_test) 

pd.crosstab(test_pred_m, y_test)

# Training Data accuracy
train_pred_m = classifier_mb.predict(x_train)
accuracy_train_m = np.mean(train_pred_m == y_train)
accuracy_train_m
