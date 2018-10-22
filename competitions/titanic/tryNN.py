# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class_idx = 6


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')


#manage Age
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(),
                                          train_dataset["Age"].mean() + train_dataset["Age"].std(),
                                          size = train_dataset["Age"].isnull().sum())

test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(),
                                          test_dataset["Age"].mean() + test_dataset["Age"].std(),
                                          size = test_dataset["Age"].isnull().sum())

train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages
train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age']    = test_dataset['Age'].astype(int)

# Embarked 
train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)
train_dataset['Port'] = train_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_dataset['Embarked']
del test_dataset['Embarked']

# Fare
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)




# Feature that tells whether a passenger had a cabin on the Titanic
train_dataset['Has_Cabin'] = train_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_dataset['Has_Cabin'] = test_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# engineer a new Title feature
# group them
full_dataset = [train_dataset, test_dataset]

##engineer the family size feature
for dataset in full_dataset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
### new try 

# Create new feature IsAlone from FamilySize
for dataset in full_dataset:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
##############################


# Get titles from the names
train_dataset['Title'] = train_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_dataset['Title'] = test_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


    
## Create new column "FamilySizeGroup" and assign "Alone", "Small" and "Big"
for dataset in full_dataset:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'

## Get the average survival rate of different FamilySizes
train_dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

for dataset in full_dataset:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        
#for dataset in full_dataset:    
#    dataset.loc[ dataset['Age'] <= 7, 'Age'] = 0
#    dataset.loc[(dataset['Age'] > 7) & (dataset['Age'] <= 14), 'Age'] = 1
#
#    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 24), 'Age'] = 2
#
#    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 32), 'Age'] = 3
#
#    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 4
#
#    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 56), 'Age'] = 5
#
#    dataset.loc[(dataset['Age'] > 56) & (dataset['Age'] <= 64), 'Age'] = 6
#
#    dataset.loc[ dataset['Age'] > 64, 'Age'] = 7
#




for dataset in full_dataset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# map the new features
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

# engineer a new  features
for dataset in full_dataset:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 1 ),'IsChildandRich'] = 1  
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 2 ),'IsChildandRich'] = 1  
    
#for dataset in full_dataset:
#    dataset['Age*Class'] = dataset.Age * dataset.Pclass 


#for dataset in full_dataset:
#    dataset['Sex*Class'] = dataset.Sex * dataset.Pclass 

#for dataset in full_dataset:
#    dataset['Sex*Age'] = dataset.Sex * dataset.Age 
    
#for dataset in full_dataset:
#    dataset['Age*Class*Sex'] = (dataset.Age * dataset.Pclass) + dataset.Sex

for data in full_dataset:
    # classify Cabin by fare
    data['Cabin'] = data['Cabin'].fillna('X')
    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
    data['Cabin'] = data['Cabin'].replace(['A', 'D', 'E', 'T'], 'M')
    data['Cabin'] = data['Cabin'].replace(['B', 'C'], 'H')
    data['Cabin'] = data['Cabin'].replace(['F', 'G'], 'L')
    data['Cabin'] = data['Cabin'].map({'X': 0, 'L': 1, 'M': 2, 'H': 3}).astype(int) 
    #data['Cabin'].loc[~data['Cabin'].isnull()] = 1
    #data['Cabin'].loc[data['Cabin'].isnull()] = 0

    
# Delete Name column from datasets (No need for them in the analysis)
del train_dataset['Name']
del test_dataset['Name']

del train_dataset['SibSp']
del test_dataset['SibSp']

del train_dataset['Parch']
del test_dataset['Parch']

del train_dataset['FamilySize']
del test_dataset['FamilySize']

#del train_dataset['FamilySizeGroup']
#del test_dataset['FamilySizeGroup']

del train_dataset['Cabin']
del test_dataset['Cabin']

# Delete Ticket column from datasets  (No need for them in the analysis)
del train_dataset['Ticket']
del test_dataset['Ticket']

del train_dataset['Port']
del test_dataset['Port']


# Cabin has a lot of nan values, so i will remove it
#del train_dataset['Cabin']
#del test_dataset['Cabin']

##title_dummies_titanic  = pd.get_dummies(train_dataset['Title'])
##train_dataset = train_dataset.join(title_dummies_titanic)
##
##title_dummies_titanic  = pd.get_dummies(test_dataset['Title'])
##test_dataset = test_dataset.join(title_dummies_titanic)
##
### Drop
##train_dataset.drop(['Title'], axis=1,inplace=True)
##test_dataset.drop(['Title'], axis=1,inplace=True)


print('----Finish data cleaning ------------')


del train_dataset['PassengerId']

#X_train = train_dataset.drop("Survived",axis=1).as_matrix()
#Y_train = train_dataset["Survived"].as_matrix()
#X_test  = test_dataset.drop("PassengerId",axis=1).copy().as_matrix()

X = train_dataset.drop("Survived",axis=1)
y = train_dataset["Survived"]
X_test  = test_dataset.drop("PassengerId",axis=1).copy()

## Importing the dataset
#dataset = train_dataset
#
#
#X = dataset.iloc[:, [2, 4,5,6,7, 9,11]].values
#y = dataset.iloc[:, 1].values
#
#
#
## Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 2].reshape(-1,1))
#X[:, 2] = imputer.transform(X[:, 2].reshape(-1,1))[:,0]


#
##X[:,5] = np.array(['0'+''.join(i for i in x if i.isdigit()) for x in X[:,5]]).astype(int)
## Encoding categorical data
## Encoding the Independent Variables
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#X[:, class_idx] = labelencoder_X.fit_transform(X[:, class_idx].astype('str'))
#onehotencoder = OneHotEncoder(categorical_features = [class_idx])
#X = onehotencoder.fit_transform(X).toarray()
#onehotencoder = OneHotEncoder(categorical_features = [class_idx])
#X = onehotencoder.fit_transform(X).toarray()
#
#X=X[:, 1:10]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


import tensorflow as tf

# Initialize placeholders for data
n = X.shape[1]
x = tf.placeholder(dtype=tf.float32, shape=[None, n])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])


# number of neurons in each layer

input_num_units = n
hidden_num_units = 50
output_num_units = 1



# Build Neural Network Weights
initializer = tf.contrib.layers.xavier_initializer()
weights = {
    'hidden1': tf.Variable(initializer([input_num_units, hidden_num_units])),
    'hidden2': tf.Variable(initializer([hidden_num_units, hidden_num_units])),
    'hidden3': tf.Variable(initializer([hidden_num_units, hidden_num_units])),
    'output': tf.Variable(initializer([hidden_num_units, output_num_units])),
}

biases = {
    'hidden1': tf.Variable(initializer([hidden_num_units])),
    'hidden2': tf.Variable(initializer([hidden_num_units])),
    'hidden3': tf.Variable(initializer([hidden_num_units])),
    'output': tf.Variable(initializer([output_num_units])),
}



# Set hyperparameters

learning_rate = 0.01
epochs = 2000


# Build model 

hidden_1_layer = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_1_layer = tf.nn.dropout(tf.nn.relu(hidden_1_layer),keep_prob = 0.6)
hidden_2_layer = tf.add(tf.matmul(hidden_1_layer, weights['hidden2']), biases['hidden2'])
hidden_2_layer = tf.nn.dropout(tf.nn.relu(hidden_2_layer),keep_prob = 0.6)
hidden_3_layer = tf.add(tf.matmul(hidden_2_layer, weights['hidden2']), biases['hidden2'])
hidden_3_layer = tf.nn.dropout(tf.nn.relu(hidden_3_layer),keep_prob = 0.6)

output_layer = tf.matmul(hidden_3_layer, weights['output']) + biases['output']



# Set loss function and goal i.e. minimize loss

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y))
opt = tf.train.AdamOptimizer(learning_rate)
goal = opt.minimize(loss)


prediction = tf.round(tf.nn.sigmoid(output_layer))
correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
recall = tf.metrics.recall(labels = y, predictions = prediction)
accuracy = tf.reduce_mean(correct)


# Initialize lists to store loss and accuracy while training the model

loss_trace = []
train_acc = []
dev_acc = []





# Start tensorflow session

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(tf.local_variables_initializer())





# training model


for epoch in range(epochs):
    sess.run(goal, feed_dict={x: X_train, y: np.matrix(y_train).T})
    
    # calculate results for epoch
    
    temp_loss = sess.run(loss, feed_dict={x: X_train, y: np.matrix(y_train).T})
    temp_train_acc = sess.run(accuracy, feed_dict={x: X_train, y: np.matrix(y_train).T})
   # temp_dev_acc = sess.run(accuracy, feed_dict={x: X_dev, y: np.matrix(y_dev).T})
    
    # save results in a list
    
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    #dev_acc.append(temp_dev_acc)
    
    # output
    
    if (epoch + 1) % 200 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} '.format(epoch + 1, temp_loss,
                                                                          temp_train_acc))



y_test_preds_nn = sess.run(prediction, feed_dict ={x: X_test})
#y_dev_preds_nn = sess.run(prediction, feed_dict ={x: X_dev})

sess.close()

