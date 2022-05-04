import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

df = pd.read_csv('data/traffic_df.csv')

sns.displot(df, x='month', hue='label', fill=True)

df['label'] = [1 if i > df.prob.median() else 0 for i in df['prob']]
df = df.drop(columns=['Unnamed: 0', 'accidents', 'prob'])
df['month'] = [str(i) for i in df.month]

enc_cols = df.select_dtypes(include='object').columns
df_final = pd.get_dummies(df, columns=enc_cols)


models=[
   ('Logistic Regression', LogisticRegression()),
   ('Ridge Classifier', RidgeClassifier()),
   ('Decision Tree', DecisionTreeClassifier()),
   ('Support Vector Machine', SVC()),
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
    kfold=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    cv_results=cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') # make_scorer(f1_score, pos_label='light_injury')
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg='%s: %f: (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)
    

grid_pars = {
    'criterion' : ['gini', 'entropy'], 
    'max_depth': [2,4,6,8,10,12], 
    'min_samples_split': range(1,10),
    'min_samples_leaf': range(1,5),
    'random_state': [42]
    }

grid = GridSearchCV(
    DecisionTreeClassifier(), 
    param_grid=grid_pars,
    cv=5,
    n_jobs=1, 
    verbose=1
    )

grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid.best_score_

grid_df = pd.DataFrame(grid.cv_results_)

grid_df

sns.


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)

confusion_matrix(y_test, y_preds)

fig = plt.figure()
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

plot_confusion_matrix(clf, X_test, y_test)