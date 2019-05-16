# data analysis and wrangling
import bisect
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cluster import KMeans


def ComputeScore(clf, X, y, scoring='accuracy'):
        xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
        return np.mean(xval) # # #


def FeatureSelections(train,targets,test):
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)

    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    print(features)

    # model = SelectFromModel(clf, prefit=True,threshold="0.2*mean")
    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(test)

    return train_reduced, test_reduced


def ModelsPerformance(train_reduced,targets):
    logreg = LogisticRegression(max_iter=400)
    logreg_cv = LogisticRegressionCV(max_iter=800)
    rf = RandomForestClassifier()
    gboost = GradientBoostingClassifier()

    models = [logreg, logreg_cv, rf, gboost]
    score_models = {}
    for model in models:
        model_name = re.sub(r"[>']","",str(model.__class__).split('.')[-1])
        print(f'Cross-validation of : {model_name}')
        score = ComputeScore(clf=model, X=train_reduced, y=targets, scoring='accuracy')
        print(f'CV score = {score}')
        print('****')
        score_models[model] = score
    best_model = max(score_models.keys(),key=lambda key: score_models[key])
    print("Best Model: " + str(re.sub(r"[>']","",str(best_model.__class__).split('.')[-1])))
    return best_model


def OptimizeModel(model,train,targets):

    parameter_grid = {
        'learning_rate': [0.1, 0.05, 0.2],
        'max_depth': [4, 6, 8, 10],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
    }

    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(model,scoring='accuracy',param_grid=parameter_grid,cv=cross_validation,verbose=1,n_jobs=-1)


    model = grid_search.fit(train, targets)
    parameters = grid_search.best_params_

    print('Best Optimize Score: {}'.format(grid_search.best_score_))
    print('Best Optimize Parameters: {}'.format(grid_search.best_params_))

    return model


def ImputeMissing(X, n_clusters=8, max_iter=10):
    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids.
            cls = KMeans(n_clusters, init=prev_centroids,n_init=1)
        else:
            cls = KMeans(n_clusters)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return pd.DataFrame(X_hat)


def CategoryToNumeric(dataframe,columns):
    for col in columns:
        labels = dataframe[col].astype('category').cat.categories.tolist()
        replace_map_comp = {col: {k: v for k, v in zip(labels, list(range(0, len(labels))))}}
        dataframe.replace(replace_map_comp, inplace=True)


def Bin(score, breakpoints, bins):
    i = bisect.bisect_left(breakpoints, score)
    return bins[i]


def AddDummies(dataframe,columns, dropers):

    df = dataframe.copy()
    for col in columns:
        dummy = pd.get_dummies(dataframe[col], prefix=col)
        df = pd.concat([df,dummy],axis=1)

    df.drop(dropers, axis=1, inplace=True)
    return df


def KaggleTest(train,targets,test,best_model):
    model = OptimizeModel(best_model,train,targets)
    prediction = model.predict(test)
    df_test = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': prediction})
    np.savetxt('Kaggle.csv', df_test, fmt='%d,%d', header='PassengerId,Survived',comments='')
    print("-I- Kaggle Results Saved.")
