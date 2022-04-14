import random
import re
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import researchpy as rp
from helper import plot_learning_curve

acc_f=os.listdir("data/accidents_data")

acc_df_l=[pd.read_csv('data/accidents_data/'+f, sep=';', decimal=',', encoding_errors='ignore').dropna() for f in acc_f]

lor=pd.read_csv('data/lor_2020.txt', sep=';')
lor_dic={int(re.split(r'(\d+ )', i)[1]):re.split(r'(\d+ )', i)[2] for i in lor['Planungsraum']}

[print(sorted(df.columns)) for df in acc_df_l]

acc_df_l[0].rename(columns={'STRZUSTAND':'USTRZUSTAND'}, inplace=True)
acc_df_l[0].rename(columns={'IstSonstig':'IstSonstige'}, inplace=True)

all_accs=pd.concat(acc_df_l)

bike_accs=all_accs[all_accs['IstRad']==1]
bike_accs['LOR']=bike_accs['LOR'].astype(int)
[print(col, bike_accs[col].isnull().values.any()) for col in bike_accs.columns]


dummy_dic={
    'BEZ':{1:'Mitte', 2:'Friedrichshain-Kreuzberg', 3:'Pankow', 4:'Charlottenburg-Wilmersdorf', 5:'Spandau', 6:'Steglitz-Zehlendorf',
           7:'Tempelhof-Schöneberg', 8:'Neukölln', 9:'Treptow-Köpenick', 10:'Marzahn-Hellersdorf', 11:'Lichtenberg', 12:'Reinickendorf'},
    'LOR':lor_dic,
    'UWOCHENTAG':{1:'sunday', 2:'monday', 3:'tuesday', 4:'wednesday', 5:'thursday', 6:'friday', 7:'saturday'},
    'UKATEGORIE':{1:'death', 2:'heavy_injury', 3:'light_injury'},
    'UART':{1:'stationary', 2:'ahead', 3:'parallel', 4:'oncoming', 5:'crossing', 6:'pedestrian', 7:'obstacle', 8:'lane_right', 9:'lane_left', 0:'other'},
    'UTYP1':{1:'driving', 2:'turn', 3: 'crossing', 4:'overriding', 5:'stationary', 6:'parallel', 7:'other'},
    'ULICHTVERH':{0:'daylight', 1:'twilight', 2:'dark'},
    'IstRad':{0:'no_bicycle', 1:'bicycle'},
    'IstPKW':{0:'no_car', 1:'car'},
    'IstFuss':{0:'no_pedestrian', 1:'pedestrian'},
    'IstKrad':{0:'no_motorcycle', 1:'motorcycle'},
    'IstGkfz':{0:'no_lorry', 1:'lorry'},
    'IstSonstige':{0:'no_other', 1:'other'},
    'USTRZUSTAND':{0:'dry', 1:'wet', 2:'icy', '0':'dry', '1':'wet', '2':'icy'}
}

for column in bike_accs.columns:
    if column in dummy_dic:
        bike_accs[column]=[dummy_dic[column][i] for i in bike_accs[column]]

bike_accs.drop(columns=['STRASSE', 'LOR_ab_2021'], inplace=True)
bike_accs['UKATEGORIE']=['light_injury' if i == 'light_injury' else 'severe_injury' for i in bike_accs['UKATEGORIE']]

bike_accs.drop(columns=['OBJECTID', 'LINREFX', 'LINREFY', 'XGCSWGS84', 'YGCSWGS84'], inplace=True)

encoder=OneHotEncoder(sparse=False)
data=encoder.fit_transform(bike_accs.loc[:, bike_accs.columns != 'UKATEGORIE'])

df_final=pd.DataFrame(data, columns=encoder.get_feature_names(input_features=bike_accs.columns[bike_accs.columns!='UKATEGORIE']))

##### Descriptive Statistics:

summary_l=[]
for column in bike_accs.columns:
    summary_l.append(rp.summary_cat(bike_accs[column]))

pd.concat(summary_l).to_csv('sum_df.csv', sep=';', decimal=',', encoding='utf-8-sig')



models=[
   ('Logistic Regression', LogisticRegression(class_weight='balanced', solver='liblinear')),
   ('Ridge Classifier', RidgeClassifier(class_weight='balanced')),
   ('Decision Tree', DecisionTreeClassifier(class_weight='balanced')),
   ('Support Vector Machine', SVC(class_weight='balanced')),
   ('Random Forest', RandomForestClassifier(class_weight='balanced')),
]

X_train, X_test, y_train, y_test=train_test_split(
    df_final,
    bike_accs['UKATEGORIE'],
    test_size=.2,
    random_state=42
)

#### Training with weighted classes

names=[]
results=[]
for name, model in models:
    kfold=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_results=cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc') # make_scorer(f1_score, pos_label='light_injury')
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)



models_sampling=[
   ('Logistic Regression', LogisticRegression(class_weight='balanced', solver='liblinear')),
   ('K-Nearest-Neighbour', KNeighborsClassifier()),
   ('Gaussian NB', GaussianNB()),
   ('Decision Tree', DecisionTreeClassifier(class_weight='balanced')),
   ('Support Vector Machine', SVC(class_weight='balanced')),
   ('Random Forest', RandomForestClassifier(class_weight='balanced')),
   ('Multinomial NB', MultinomialNB())   
]


###### SMOTE oversampling

smote=SMOTE()
X_train_smote, y_train_smote=smote.fit_resample(X_train, y_train)

names_smote=[]
results_smote=[]
for name, model in models_sampling:
    kfold=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_results=cross_val_score(model, X_train_smote, y_train_smote, cv=kfold, scoring='roc_auc')
    print(cv_results)
    results_smote.append(cv_results)
    names_smote.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)


##### Undersampling 

ru=RandomUnderSampler()
X_train_ru, y_train_ru = ru.fit_resample(X_train, y_train)



names_ru=[]
results_ru=[]
for name, model in models_sampling:
    kfold=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_results=cross_val_score(model, X_train_ru, y_train_ru, cv=kfold, scoring='roc_auc')
    print(cv_results)
    results_ru.append(cv_results)
    names_ru.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)



weights={'severe_injury':1.8, 'light_injury':0.6}

clf=SVC(class_weight='balanced')
clf.fit(X_train_smote, y_train_smote)

y_preds=clf.predict(X_test)

confusion_matrix(y_test, y_preds)

fig=plt.figure()
ax=fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



###### Testing RF SMOTE + plots

clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train_smote, y_train_smote)
y_pred = clf.predict(X_test)


plot_confusion_matrix(clf, X_test, y_test) 


plot_learning_curve(estimator=RandomForestClassifier(class_weight='balanced'), 
                    title='Learning Curve of RF (SMOTE)',
                    X=X_train_smote, y=y_train_smote,
                    cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
                    )
plt.show()
