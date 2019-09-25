#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 03:19:31 2017

@author: junshuaizhang
"""

import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
import sklearn as sk

train = pd.read_csv('/Users/junshuaizhang/Kaggle/titanic/train-3.csv')
test = pd.read_csv('/Users/junshuaizhang/Kaggle/titanic/test-2.csv')
real_data = pd.read_csv('/Users/junshuaizhang/Kaggle/titanic/titanic3_csv.csv')

test.insert(1, 'Survived', -1)

##union two data set into one.
all_data = pd.concat([train,test])

#scatter_matrix(train[['Survived','Age','Pclass']],
#               alpha=0.9, figsize=(10, 10), diagonal='hist',
#               s=1)

# Change the value of  Pclass column
# 1->3, 3->1
all_data.Pclass.replace([1,3],[3,1],inplace=True)

##Extract title from Name column.
P=all_data.Name.str.extract('[, ](?P<Title>[a-zA-z]+)[.]',
                          expand=True)
all_data.insert(4,'Title',P)

##function used to change sex column into number 0,1
def func(row):
    if row['Sex'] == 'male':
        return 0
    elif row['Sex'] =='female':
        return 1
    else:
        return None
##Change sex column into number 0,1
all_data.insert(6,'Sex_number',
                all_data.apply(func, axis=1))

prefix=all_data.Ticket.str.extract('(?P<Ticket_Prefix>[a-zA-z0-9,./]+)[ ]',
                          expand=True)
prefix=prefix.Ticket_Prefix.str.replace('.','')
prefix=prefix.str.replace('/','')
prefix = prefix.str.extract('(?P<Ticket_Prefix>[A-Za-z]{1,3})',expand=True)
all_data.insert(11, 'Ticket_Prefix',prefix)
ticket_number = all_data.Ticket.str.extract('(?P<Ticket_Number>[0-9]{4,10})',
                          expand=True)
all_data.insert(12,'Ticket_Number', ticket_number)
all_data.Cabin.fillna('',inplace=True)
Cabin_prefix = all_data.Cabin.str.extract('(?P<Cabin_Category>[A-Z]{1})',expand=True)
Cabin_number = all_data.Cabin.str.extract('(?P<Cabin_number>[0-9]+)',expand=True)
all_data.insert(15,'Cabin_prefix', Cabin_prefix)
all_data.insert(16,'Cabin_number', Cabin_number)


#treat some columns with only few missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
imp2=Imputer(missing_values='NaN', strategy='mean', axis=1)

Ticke_number_filled = imp2.fit(all_data['Ticket_Number'].values.reshape(-1,1))
Ticke_number_filled = Ticke_number_filled.transform(all_data['Ticket_Number'])
Ticke_number_filled = pd.DataFrame(Ticke_number_filled.reshape(all_data.shape[0],1))
all_data.insert(13,'Ticket_Number_filled',Ticke_number_filled)

embark1 = imp.fit(all_data['Embarked'].values.reshape(-1,1))
#embark1 = imp.transform(all_data['Embarked'])



##Encoding categorical features
le = sk.preprocessing.LabelEncoder()

## impute missing data with numeric values.
def ImputeNumeric(data,column_name,imputeStrategy):
    from sklearn.preprocessing import Imputer
    imp=Imputer(missing_values='NaN', strategy=imputeStrategy, axis=1)
    imp.fit(data[column_name].values.reshape(-1,1))
    column_filled = imp.transform(data[column_name])
    column_filled = pd.DataFrame(column_filled.reshape(all_data.shape[0],1))
    data.insert(all_data.shape[1],'{}_filled'.format(column_name),column_filled)
    return data

all_data = ImputeNumeric(all_data,'Fare','mean')
    


## Treat the missing attribute with non numberic values. convert it into numbercal lebel
## but not fill the missing cells.
def labelTextCategoricalAttr(data,key,column, has_nan=True,need_fill=True):
    import pandas as pd
    import sklearn as sk
    le = sk.preprocessing.LabelEncoder()
    #enc = sk.preprocessing.OneHotEncoder()
    ## get data with key and objective column with none-null values.
    if has_nan == True and need_fill==True:
        from sklearn.preprocessing import Imputer
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
        
        sub_data = data[data[column].notnull()][[key,column]]
        le = le.fit(sub_data[column])
        sub_data2 = le.transform(sub_data[column])
        sub_data.insert(0,'{}_numeric'.format(column),sub_data2)
        sub_data.pop(column)
        data=pd.merge(data,sub_data,on=key,how='left')
        column_filled = imp.fit(data['{}_numeric'.format(column)].values.reshape(-1,1))
        column_filled = imp.transform(data['{}_numeric'.format(column)])
        column_filled = pd.DataFrame(column_filled.reshape(data.shape[0],1))
        data.pop('{}_numeric'.format(column))
        data.insert(data.shape[1],'{}_numeric'.format(column),column_filled )
    
    elif has_nan == True and need_fill==False:
        #from sklearn.preprocessing import Imputer
        #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
        
        sub_data = data[data[column].notnull()][[key,column]]
        le = le.fit(sub_data[column])
        sub_data2 = le.transform(sub_data[column])
        sub_data.insert(0,'{}_numeric'.format(column),sub_data2)
        sub_data.pop(column)
        data=pd.merge(data,sub_data,on=key,how='left')
#        column_filled = imp.fit(data['{}_numeric'.format(column)].values.reshape(-1,1))
#        column_filled = imp.transform(data['{}_numeric'.format(column)])
#        column_filled = pd.DataFrame(column_filled.reshape(data.shape[0],1))
#        data.pop('{}_numeric'.format(column))
#        data.insert(data.shape[1],'{}_numeric'.format(column),column_filled )
        
    else:
        sub_data = data[[key,column]]
        le = le.fit(sub_data[column])
        sub_data2 = le.transform(sub_data[column])
        sub_data.insert(0,'{}_numeric'.format(column),sub_data2)
        sub_data.pop(column)
        data=pd.merge(data,sub_data,on=key,how='left')
    
    
    return data


    
            

#embark = all_data[all_data['Embarked'].notnull()][['PassengerId','Embarked']]
#le = le.fit(embark['Embarked'])
#embark1 = le.transform(embark['Embarked'])
#embark.insert(0,'Embarked_numeric',embark1)
#embark.pop('Embarked')
#all_data=pd.merge(all_data,embark,on='PassengerId',how='left')

all_data = labelTextCategoricalAttr(all_data, 'PassengerId','Embarked', has_nan=True)
all_data = labelTextCategoricalAttr(all_data, 'PassengerId','Title', has_nan=True)
all_data = labelTextCategoricalAttr(all_data, 'PassengerId','Cabin_prefix', has_nan=True, need_fill = False)



# do one hot encoding and add the result into the original data frame.
# data must have no np.nan value.
# can use labelTextCategoricalAttr() or imputer to treat np.nan values.
def oneHotEncodingPlus(data, column):
    import sklearn as sk
    enc = sk.preprocessing.OneHotEncoder()
    p = enc.fit_transform(data[column].values.reshape(-1,1)).toarray()
    p = pd.DataFrame(p)
    p.columns = ['{}_c{}'.format(column,i) for i in range(len(p.columns))]
    data = pd.concat([data,p],axis=1)
    return data

all_data=oneHotEncodingPlus(all_data, 'Title_numeric')
all_data = oneHotEncodingPlus(all_data, 'Embarked_numeric')




# Scale the numerical data with min_max_scaler
def min_max_scaling(data,column):
    import sklearn as sk
    min_max_scaler = sk.preprocessing.MinMaxScaler()
    X_train_minmax = pd.DataFrame(min_max_scaler.fit_transform(data[column]))
    data.insert(data.shape[1],'{}_scaled'.format(column),X_train_minmax)
    return data

# Scale the numerical data with min_max_scaler
def standard_scaling(data,column):
    import sklearn as sk
    X_scaled = sk.preprocessing.scale(data[column])
    #X_scaled = pd.DataFrame(scaler.fit_transform(data[column]))
    data.insert(data.shape[1],'{}_scaled'.format(column),X_scaled)
    return data

#all_data = standard_scaling(all_data,'Fare_filled')
all_data = standard_scaling(all_data,'Fare_filled')
all_data = standard_scaling(all_data,'SibSp')
all_data = standard_scaling(all_data,'Parch')
all_data = standard_scaling(all_data,'Ticket_Number_filled')
all_data = standard_scaling(all_data,'Pclass')



##feature selection
train = all_data[all_data['Survived']!=-1 ]
train = train[train['Age'].notnull()]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
S = SelectKBest(f_classif, k=1).fit(train[['Pclass_scaled','Sex_number','SibSp','Parch','Fare_filled_scaled','Ticket_Number_filled_scaled']],train['Survived'])
cabin_test = all_data[all_data['Cabin_number'].notnull()]
S2 = SelectKBest(f_classif, k=1).fit(cabin_test[['Cabin_number']],cabin_test['Survived'])
#print(S.pvalues_)



#age_data = all_data[all_data['Age'].notnull()]
#non_age_data = all_data[all_data['Age'].isnull()]
#age_cabin_pred_data = age_data[['PassengerId', 'Survived', 'Age','Cabin_prefix_numeric',
#       'Sex_number',   'Title_numeric_c0',
#       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
#       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
#       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
#       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
#       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
#       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
#       'Embarked_numeric_c1', 'Embarked_numeric_c2', 
#       'Fare_filled_scaled', 'SibSp_scaled', 'Parch_scaled',
#       'Ticket_Number_filled_scaled', 'Pclass_scaled']]

all_age_cabin_pred_data = all_data[['PassengerId', 'Survived', 'Age','Cabin_prefix_numeric',
       'Sex_number',   'Title_numeric_c0',
       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
       'Embarked_numeric_c1', 'Embarked_numeric_c2', 
       'Fare_filled_scaled', 'SibSp_scaled', 'Parch_scaled',
       'Ticket_Number_filled_scaled', 'Pclass_scaled']]



#cabin_prefix_data = all_data[all_data['Cabin_prefix_numeric'].notnull()]
#non_cabinprefix_data = all_data[all_data['Cabin_prefix_numeric'].isnull()]
#cabinprefix_pred_data = cabin_prefix_data[['PassengerId', 'Survived', 'Cabin_prefix_numeric',
#       'Sex_number',   'Title_numeric_c0',
#       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
#       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
#       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
#       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
#       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
#       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
#       'Embarked_numeric_c1', 'Embarked_numeric_c2', 
#       'Fare_filled_scaled', 'SibSp_scaled', 'Parch_scaled',
#       'Ticket_Number_filled_scaled', 'Pclass_scaled']]
#
#cabinprefix_pred_data = all_data[['PassengerId', 'Survived', 'Cabin_prefix_numeric',
#       'Sex_number',   'Title_numeric_c0',
#       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
#       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
#       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
#       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
#       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
#       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
#       'Embarked_numeric_c1', 'Embarked_numeric_c2', 
#       'Fare_filled_scaled', 'SibSp_scaled', 'Parch_scaled',
#       'Ticket_Number_filled_scaled', 'Pclass_scaled']]



from sklearn.decomposition import PCA
p = PCA(n_components=20)
#p_cabin = PCA(n_components=7)
x_pca = p.fit_transform(all_age_cabin_pred_data.loc[:,'Sex_number':'Pclass_scaled'])
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['x_pca_{}'.format(i) for i in range(x_pca.shape[1])]
all_age_cabin_pred_data=pd.concat([all_age_cabin_pred_data,x_pca],axis=1)
age_pred_data = all_age_cabin_pred_data[all_age_cabin_pred_data['Age'].notnull()]
non_age_data = all_age_cabin_pred_data[all_age_cabin_pred_data['Age'].isnull()]
non_X_PCA = non_age_data.loc[:,'x_pca_0':'x_pca_15']
X_PCA = age_pred_data.loc[:,'x_pca_0':'x_pca_15']
cabinprefix_pred_data = all_age_cabin_pred_data[all_age_cabin_pred_data['Cabin_prefix_numeric'].notnull()]
non_cabinprefix_pred_data = all_age_cabin_pred_data[all_age_cabin_pred_data['Cabin_prefix_numeric'].isnull()]
cabin_pca = cabinprefix_pred_data.loc[:,'x_pca_0':'x_pca_7']
non_cabin_pca = non_cabinprefix_pred_data.loc[:,'x_pca_0':'x_pca_7']
#cabin_pca = p_cabin.fit_transform(all_age_cabin_pred_data.loc[:,'Sex_number':'Pclass_scaled'])

# draw 3-D plot of PCA
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(1, figsize=(8, 6))
#ax = Axes3D(fig, elev=-150, azim=110)
#ax.scatter(X_PCA[:, 0], X_PCA[:, 1], X_PCA[:, 2], c=age_pred_data['Age'],
#           cmap=plt.cm.Paired)
#plt.show()

#test prediction
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn import linear_model
#reg = linear_model.Lasso(alpha = 0.01)
#reg = linear_model.ElasticNet(alpha=0.0001, l1_ratio=0.5)
reg = linear_model.Ridge (alpha = .05)
from sklearn import svm
reg = svm.SVR(C=1000,kernel='rbf')
clf = svm.LinearSVC(C=3)
scores_age = cross_val_score(
    reg, X_PCA, age_pred_data['Age'], cv=10, scoring='neg_mean_squared_error')
scores_cabin = cross_val_score(
    clf, cabin_pca, cabinprefix_pred_data['Cabin_prefix_numeric'], cv=10, scoring='accuracy')
#
reg.fit(X_PCA[0:800],age_pred_data['Age'][0:800])
y_pred = reg.predict(X_PCA[800:])
y_pred = pd.DataFrame(y_pred)
y = pd.DataFrame(np.array(age_pred_data['Age'][800:]))
y.columns=['y']
combined_y = pd.concat([y,y_pred], axis=1)
#
clf.fit(cabin_pca[0:200],cabinprefix_pred_data['Cabin_prefix_numeric'][0:200])
cabin_pred = clf.predict(cabin_pca[200:])
cabin_pred = pd.DataFrame(cabin_pred)
cabin = pd.DataFrame(np.array(cabinprefix_pred_data['Cabin_prefix_numeric'][200:]))
cabin.columns=['cabin']
combined_cabin = pd.concat([cabin,cabin_pred], axis=1)


##predict the missing values and combine them with non-nan values
# then insert new column into the original data set.
# predict Age
reg.fit(X_PCA,age_pred_data['Age'])
age_predicted = reg.predict(non_X_PCA)
age_predicted = pd.DataFrame(age_predicted)
age_predicted.columns=['Age']
non_age_data = non_age_data['PassengerId'].reset_index()
non_age_data.pop('index')
non_age_data = pd.concat([non_age_data, age_predicted], axis = 1)
age_pred_data = age_pred_data[['PassengerId', 'Age']]
age_data = pd.concat([age_pred_data,non_age_data],ignore_index = True)
age_data = age_data.sort_values(['PassengerId'])
age_data = age_data.reset_index()
age_data.pop('index')

all_data.insert(8,'Age_Predicted',age_data['Age'])

#predict cabin prefix
clf = svm.LinearSVC(C=3)
clf.fit(cabin_pca,cabinprefix_pred_data['Cabin_prefix_numeric'])
cabin_predicted = clf.predict(non_cabin_pca)
cabin_predicted = pd.DataFrame(cabin_predicted)
cabin_predicted.columns=['Cabin_prefix_numeric']
non_cabinprefix_pred_data = non_cabinprefix_pred_data['PassengerId'].reset_index()
non_cabinprefix_pred_data.pop('index')
non_cabinprefix_pred_data = pd.concat([non_cabinprefix_pred_data, cabin_predicted], axis = 1)
cabinprefix_pred_data = cabinprefix_pred_data[['PassengerId', 'Cabin_prefix_numeric']]
cabinprefix_data = pd.concat([cabinprefix_pred_data,non_cabinprefix_pred_data],ignore_index = True)
cabinprefix_data = cabinprefix_data.sort_values(['PassengerId'])
cabinprefix_data = cabinprefix_data.reset_index()
cabinprefix_data.pop('index')

all_data.insert(24,'Cabin_prefix_numeric_Predicted',cabinprefix_data['Cabin_prefix_numeric'])


all_data = standard_scaling(all_data,'Age_Predicted')
all_data = oneHotEncodingPlus(all_data,'Cabin_prefix_numeric_Predicted')


real_survived = real_data[['ticket', 'name','embarked', 'survived']]
real_survived.columns = ['Ticket','Name','Embarked', 'Survived_real']

##discretize age column.
def func2(row):
    if row['Age_Predicted'] <= 8:
        return 0
    elif row['Age_Predicted'] >8 and row['Age_Predicted'] <=25:
        return 1
    elif row['Age_Predicted'] >25 and row['Age_Predicted'] <=35:
        return 2
    elif row['Age_Predicted'] >35 and row['Age_Predicted'] <=50:
        return 3
    elif row['Age_Predicted'] >50:
        return 4
    else:
        return None
##Change sex column into number 0,1
all_data.insert(10,'Age_discretized',
                all_data.apply(func2, axis=1))

all_data = oneHotEncodingPlus(all_data, 'Age_discretized')

def func3(row):
    if row['Fare_filled'] <= 20:
        return 0
    elif row['Fare_filled'] >20 and row['Fare_filled'] <=50:
        return 1
    elif row['Fare_filled'] >50 and row['Fare_filled'] <=100:
        return 2
    elif row['Fare_filled'] >100:
        return 3
    else:
        return None
#Change sex column into number 0,1
all_data.insert(18,'Fare_discredized',
                all_data.apply(func3, axis=1))

all_data = oneHotEncodingPlus(all_data, 'Fare_discredized')
#all_data = oneHotEncodingPlus(all_data, 'Cabin_prefix_discredized')

print(all_data.columns)


all_data=pd.merge(all_data, real_survived, on=['Ticket', 'Name'] , how='left')

all_data = all_data[['PassengerId', 'Survived', 'Survived_real','Pclass', 
       'Sex_number', 'Age', 'Age_Predicted','Age_discretized', 'SibSp', 'Parch', 
       'Ticket_Prefix', 'Ticket_Number', 'Ticket_Number_filled',
       'Cabin', 'Cabin_prefix', 'Cabin_number', 'Embarked_x', 'Fare_filled',
       'Embarked_numeric', 'Title_numeric', 'Cabin_prefix_numeric',
       'Cabin_prefix_numeric_Predicted', 'Title_numeric_c0',
       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
       'Embarked_numeric_c1', 'Embarked_numeric_c2', #'Fare_filled_scaled',
       'SibSp_scaled', 'Parch_scaled', 'Ticket_Number_filled_scaled',
       'Pclass_scaled', 
       'Age_discretized_c0',
       'Age_discretized_c1', 'Age_discretized_c2', 'Age_discretized_c3',
       'Age_discretized_c4',
       'Fare_discredized_c1',
       'Fare_discredized_c2', 'Fare_discredized_c3',
#        'Cabin_prefix_discredized_c0', 'Cabin_prefix_discredized_c1',
#       'Cabin_prefix_discredized_c2',
       'Cabin_prefix_numeric_Predicted_c0',
       'Cabin_prefix_numeric_Predicted_c1',
       'Cabin_prefix_numeric_Predicted_c2',
       'Cabin_prefix_numeric_Predicted_c3',
       'Cabin_prefix_numeric_Predicted_c4',
       'Cabin_prefix_numeric_Predicted_c5',
       'Cabin_prefix_numeric_Predicted_c6',
       'Cabin_prefix_numeric_Predicted_c7'
       ]]


#train = all_data[all_data['Survived']!=-1]
train_test = all_data[['PassengerId', 'Survived', 
'Sex_number', 'Title_numeric_c0',
       'Title_numeric_c1', 'Title_numeric_c2', 'Title_numeric_c3',
       'Title_numeric_c4', 'Title_numeric_c5', 'Title_numeric_c6',
       'Title_numeric_c7', 'Title_numeric_c8', 'Title_numeric_c9',
       'Title_numeric_c10', 'Title_numeric_c11', 'Title_numeric_c12',
       'Title_numeric_c13', 'Title_numeric_c14', 'Title_numeric_c15',
       'Title_numeric_c16', 'Title_numeric_c17', 'Embarked_numeric_c0',
       'Embarked_numeric_c1', 'Embarked_numeric_c2', #'Fare_filled_scaled',
       'SibSp_scaled', 'Parch_scaled', 'Ticket_Number_filled_scaled',
       'Pclass_scaled', 
              'Age_discretized_c0',
       'Age_discretized_c1', 'Age_discretized_c2', 'Age_discretized_c3',
       'Age_discretized_c4',
       'Fare_discredized_c1',
       'Fare_discredized_c2', 'Fare_discredized_c3',
        #       'Cabin_prefix_discredized_c0', 'Cabin_prefix_discredized_c1',
       #'Cabin_prefix_discredized_c2'
       'Cabin_prefix_numeric_Predicted_c0',
       'Cabin_prefix_numeric_Predicted_c1',
       'Cabin_prefix_numeric_Predicted_c2',
       'Cabin_prefix_numeric_Predicted_c3',
       'Cabin_prefix_numeric_Predicted_c4',
       'Cabin_prefix_numeric_Predicted_c5',
       'Cabin_prefix_numeric_Predicted_c6',
       'Cabin_prefix_numeric_Predicted_c7'
       ]]
#train = train_test[all_data['Survived'] != -1]
#test  = train_test[all_data['Survived'] == -1]

p2 = PCA(n_components=30)
#p_cabin = PCA(n_components=7)
train_test_pca = p2.fit_transform(train_test.loc[:,'Sex_number':'Cabin_prefix_numeric_Predicted_c7'])
train_test_pca = pd.DataFrame(train_test_pca)
train_test_pca.columns=['train_test_pca_{}'.format(i) for i in range(train_test_pca.shape[1])]
train_test = pd.concat([train_test,train_test_pca],axis=1)

train = train_test[train_test['Survived'] != -1]
train_pca = train.loc[:,'train_test_pca_0':'train_test_pca_20']
test = train_test[train_test['Survived'] == -1]
test_pca = test.loc[:,'train_test_pca_0':'train_test_pca_20']


clf2 = svm.LinearSVC(C=1)
#from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#clf = sk.neighbors.KNeighborsClassifier(9, weights='uniform')

clf5 = BaggingClassifier(sk.neighbors.KNeighborsClassifier(55, weights='uniform'),
                            max_samples=0.7, max_features=0.7, n_estimators=100)
boost = AdaBoostClassifier(LogisticRegression(C=1.7),n_estimators=100)

parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50,10,100],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [2, 5, 10],
                 'bootstrap': [True, False],
                 }
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(n_estimators=100)
gridsearch = GridSearchCV(rfc, parameter_grid, cv=5,
                       scoring='accuracy')

#gridsearch.fit(train_pca, train['Survived'])

#scores_train = cross_val_score(
#    clf2, train_pca, train['Survived'], cv=5, scoring='accuracy')


clf3 = sk.neighbors.KNeighborsClassifier(54, weights='uniform')
clf4 = LogisticRegression(C=0.1)
params = {'bootstrap': False,
 'max_depth': 2,
 'max_features': 'auto',
 'min_samples_leaf': 7,
 'min_samples_split': 3,
 'n_estimators': 500}
clf = RandomForestClassifier(**params)

scores_train = cross_val_score(
    clf3, train_pca, train['Survived'], cv=10, scoring='accuracy')



clf3.fit(train_pca,train['Survived'])
y = clf3.predict(test_pca)
y = pd.DataFrame(y)
y.columns = ['Survived']

test = test.reset_index()
resultid = test['PassengerId']
report = pd.concat([resultid,y],axis = 1, ignore_index=True)
report.columns= ['PassengerId','Survived']
all_data = pd.merge(all_data,report,on='PassengerId',how='left')
compare = all_data[all_data['Survived_x'] == -1][['Survived_real','Survived_y']]
compare = compare[compare['Survived_real'].notnull()]
from sklearn.metrics import accuracy_score
acc = accuracy_score(compare['Survived_real'], compare['Survived_y'])

report.to_csv('/Users/junshuaizhang/Kaggle/titanic/result.csv')


##plot stacked histogram
#plt.hist([all_data_2[all_data_2['Survived_real']==1]['Fare_filled'], all_data_2[all_data_2['Survived_real']==0]['Fare_filled']], stacked=True, color = ['g','r'],
#         bins = 100,label = ['Survived','Dead'])
#

 #draw 3-D plot of PCA
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(1, figsize=(8, 6))
#ax = Axes3D(fig, elev=-150, azim=110)
#ax.scatter(train_pca['train_test_pca_0'], train_pca['train_test_pca_1'], train_pca['train_test_pca_2'], c=train['Survived'],
#           cmap=plt.cm.Paired)
#plt.show()


#print(np.mean(scores_train))
