from pickletools import optimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import graphviz
from sklearn import estimators
from sklearn import tree
import numpy as np
from helper import plot_learning_curve
from helper import plot_search_results
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
import scikitplot as skplt
from dtreeviz.trees import dtreeviz


df = pd.read_csv('data/traffic_df2.csv')

sns.displot(df, y='BEZIRK', rug=True, rug_kws={'height':.01, 'expand_margins': True})

df['label'] = [1 if i > df.prob.median() else 0 for i in df['prob']]
df = df.drop(columns=['Unnamed: 0', 'accidents', 'prob'])
df['month'] = [str(i) for i in df.month]

enc_cols = df.select_dtypes(include='object').columns
df_final = pd.get_dummies(df, columns=enc_cols)


models=[
   ('Logistic', LogisticRegression(solver='newton-cg')),
   ('Ridge', RidgeClassifier()),
   ('Decision Tree', DecisionTreeClassifier()),
   ('SVM', SVC()),
   ('Random Forest', RandomForestClassifier()),
]


X_train, X_test, y_train, y_test=train_test_split(
    df_final.loc[:, df_final.columns != 'label'],
    df_final['label'],
    test_size=.2,
    random_state=42
)

names=[]
results=[]
for name, model in models:
    kfold=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=24)
    cv_results=cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') # make_scorer(f1_score, pos_label='light_injury')
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

### grid search

grid_pars = { 
    'max_depth': [2,4,6,8,10,12], 
    'min_samples_split': range(2,11),
    'min_samples_leaf': range(1,5),
    'criterion' : ['entropy', 'gini']
    }

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42), 
    param_grid=grid_pars,
    cv=5,
    n_jobs=1, 
    verbose=1,
    return_train_score=True
    )

grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid.best_score_

grid_df = pd.DataFrame(grid.cv_results_)

grid_df

plot_search_results(grid)

# construct best model, apply to test set

clf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=3, min_samples_split=9)
clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)


viz = dtreeviz(clf, X_test, y_test,
               target_name='target',
               feature_names=X_test.columns)



textrep = tree.export_text(clf)

confusion_matrix(y_test, y_preds)


plot_confusion_matrix(clf, X_test, y_test)


plot_learning_curve(estimator=clf, 
                    title='Learning Curve',
                    X=X_train, y=y_train,
                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
                    )
plt.show()


feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')


y_proba = clf.predict_proba(X_test)
y_pred = np.where(y_proba[:,1] > 0.5, 1, 0)

skplt.metrics.plot_roc(y_test, y_proba, title = 'ROC Plot')
skplt.metrics.plot_precision_recall(y_test, y_proba, title = 'PR Curve')


skplt.metrics.plot_cumulative_gain(y_test, y_proba, title = 'Cumulative Gains Chart')

# comparison with Random Forest
rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
y_rfc_proba = rfc_model.predict_proba(X_test)

probas_list = [y_proba, y_rfc_proba]
clf_names = ['DT', 'RF']
skplt.metrics.plot_calibration_curve(y_test, probas_list = probas_list, clf_names = clf_names)



#### subset of center districts ####
df = pd.read_csv('data/traffic_df2.csv')

sns.displot(df, y='BEZIRK', rug=True, rug_kws={'height':.01, 'expand_margins': True})

df['label'] = [1 if i > df.prob.median() else 0 for i in df['prob']]
df = df.drop(columns=['Unnamed: 0', 'accidents', 'prob'])
df['month'] = [str(i) for i in df.month]

df = df[df.BEZIRK.isin(['Mitte', 'Friedrichshain-Kreuzberg', 'Charlottenburg-Wilmersdorf', 'Neukölln', 'Tempelhof-Schöneberg', 'Pankow'])]
enc_cols = df.select_dtypes(include='object').columns
df_final = pd.get_dummies(df, columns=enc_cols)


models=[
   ('Logistic', LogisticRegression(solver='newton-cg')),
   ('Ridge', RidgeClassifier()),
   ('Decision Tree', DecisionTreeClassifier()),
   ('SVM', SVC()),
   ('Random Forest', RandomForestClassifier()),
]


X_train, X_test, y_train, y_test=train_test_split(
    df_final.loc[:, df_final.columns != 'label'],
    df_final['label'],
    test_size=.2,
    random_state=42
)

names=[]
results=[]
for name, model in models:
    kfold=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=24)
    cv_results=cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') # make_scorer(f1_score, pos_label='light_injury')
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()