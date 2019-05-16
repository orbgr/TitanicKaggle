# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import bisect
import pickle

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



split = 10
count = acc = recall = precision = max_acc = 0



def main():
    # input
    titanic_data = pd.read_csv('./DataTitanic.csv')
    test = pd.read_csv('./test.csv')

    # data preparation
    titanic_data = Cleansing(titanic_data)
    titanic_data = Blending(titanic_data)
    titanic_data = ToFloat(titanic_data)
    print("Data Sample:\n", titanic_data.head(10))
    print(titanic_data.columns)
    # classifies
    kf, titanic_explain, titanic_target, clf = InitModel(titanic_data, splits=split)
    for train_index,test_index in kf.split(titanic_explain,titanic_target):
        prediction,model = Predict(train_index,test_index, titanic_explain, titanic_target,clf)
        best_model = Results(prediction,titanic_target,test_index,model)


    print("Average Accuracy: ", 100*acc/split)
    print("Average Recall: ", recall/2*split)
    print("Average Precision: ", precision/2*split)
    titanic_data = Cleansing(test)
    titanic_data = Blending(test)
    titanic_data = ToFloat(titanic_data)
    #KaggleTest(best_mode,test)


def FeatureImportance(clf,train,y_train):
    global count
    # if (count == 0):
    #     features = pd.DataFrame()
    #     features['feature'] = train.columns
    #     features['importance'] = clf.feature_importances_
    #     features.sort_values(by=['importance'], ascending=True, inplace=True)
    #     features.set_index('feature', inplace=True)
    #     features.plot(kind='barh', figsize=(25, 25))
    #     print(features)
    #     count += 1
    #     plt.show()
    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(y_train)
    return model,train_reduced,test_reduced



def KaggleTest(best_model,test):
    prediction = best_model.predict(test)
    df_test = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': prediction})
    np.savetxt('Kaggle.csv', df_test, fmt='%d,%d', header='PassengerId,Survived')

def Results(prediction,titanic_target,test_index,model):
    global split,acc,recall,precision,max_acc,max_model

    c_matrix = confusion_matrix(titanic_target.loc[test_index], prediction)

    model_accuracy = c_matrix.trace() / c_matrix.sum()
    if (max_acc < model_accuracy):
        max_acc = model_accuracy
        max_model = model

    acc += model_accuracy

    target_values = pd.unique(titanic_target)
    for i in range(0, len(target_values)):
        recall += c_matrix[i, i] / c_matrix[i, :].sum()
        precision += c_matrix[i, i] / c_matrix[:, i].sum()

    return model


def Predict(train_index,test_index,titanic_explain,titanic_target,clf):
    X_train, X_test = titanic_explain.loc[train_index,:], titanic_explain.loc[test_index,:]
    y_train, y_test = titanic_target.loc[train_index], titanic_target.loc[test_index]
    clf = clf.fit(X_train, y_train)
    model, train_reduced, test_reduced = FeatureImportance(clf,X_train, y_train, )
    prediction = clf.predict(train_reduced)
    return prediction,model

def InitModel(titanic_data, splits):
    titanic_explain = titanic_data.drop("Survived", axis=1)
    titanic_target = titanic_data.Survived
    kf = KFold(n_splits=splits,shuffle=True, random_state=0)
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    # clf = DecisionTreeClassifier(random_state=0, criterion="gini", min_samples_leaf=5, max_depth=10)
    return kf,titanic_explain,titanic_target,clf

def Blending(titanic):
    titanic_data = titanic.copy()
    # +FamilySize   -SibSp  -Parch
    titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
    titanic_data.drop(['SibSp','Parch'], axis=1, inplace=True)

    # +Title as number  -Name
    pattern = r'.*,\s(.*)\.(.+)'
    titanic_data['Name'] = titanic_data['Name'].str.extract(pattern, expand=True)
    titanic_data['Name'] = titanic_data['Name'].map(
        {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
        } ).fillna(0)
    titanic_data.rename(columns={'Name':'Title'}, inplace=True)

    grouped_train = titanic_data.groupby(['Sex', 'Pclass', 'Title']).median().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]  # pivoting on Sex,Class,Title to median Age
    titanic_data['Age'] = titanic_data.apply(lambda row: FillAge(row,grouped_train) if np.isnan(row['Age']) else row['Age'], axis=1)

    # +TicketAlphaBet extract \D from tickets
    titanic_data['TicketAlphaBet'] = titanic_data['Ticket'].replace('[\d\./]+', '', regex=True)
    titanic_data['TicketAlphaBet'] =  titanic_data['TicketAlphaBet'].apply(lambda row: row[0] if len(row) > 0 else '?')

    # +TicketAmount a unique tickets as number -Ticket
    titanic_data['Ticket'] = titanic_data['Ticket'].replace('\D', '', regex=True)
    titanic_data['Ticket'] = titanic_data.groupby('Ticket')['Ticket'].transform('count')
    titanic_data['Ticket'] = [Bin(ticket,breakpoints=[1,3,5,100],bins='1234') for ticket in titanic_data['Ticket']]
    titanic_data['Ticket'] = titanic_data['Ticket']
    titanic_data.rename(columns={'Ticket': 'TicketAmount'}, inplace=True)


    # +Poor for some Cabin and Fare
    # titanic_data.loc[titanic_data['Fare'] < 15 & (titanic_data['Cabin'].isin(["F","G","T","E"])),"Poor"] = 1
    # titanic_data.loc[np.isnan(titanic_data['Poor']), "Poor"] = 0

    # +VIP for some Cabin and Fare
    # titanic_data.loc[titanic_data['Fare'] > 90 & (titanic_data['Cabin'].isin(["A", "B"])), "VIP"] = 1
    # titanic_data.loc[np.isnan(titanic_data['VIP']), "VIP"] = 0

    # map to number
    titanic_data['Sex'] = titanic_data['Sex'].map( {'female': 0, 'male': 1} )
    titanic_data['Embarked'] = titanic_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

    # binning
    titanic_data['Fare'] = [Bin(size, breakpoints=[8, 15, 31, float("inf")], bins='12345') for size in titanic_data['Fare']]
    titanic_data['Age'] = [Bin(age,breakpoints=[10, 20, 50, float("inf")],bins='12345') for age in titanic_data['Age']]
    titanic_data['FamilyGroupSize'] = [Bin(size, breakpoints=[1, 3, 5, float("inf")], bins='12345') for size in titanic_data['FamilySize']]

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(titanic_data['Title'], prefix='Title')
    cabin_dummies = pd.get_dummies(titanic_data['Cabin'], prefix='Cabin')
    ticketAlphaBet_dummies= pd.get_dummies(titanic_data['TicketAlphaBet'], prefix='TicketAlphaBet')
    titanic_data = pd.concat([titanic_data, titles_dummies,cabin_dummies,ticketAlphaBet_dummies], axis=1)

    titanic_data.drop(['Cabin','Title','TicketAlphaBet'], axis=1, inplace=True)

    return titanic_data



def Cleansing(titanic_data):
    titanic_copy = titanic_data.copy()
    titanic_copy.drop('PassengerId', axis=1, inplace=True)

    freq_port = titanic_copy.Embarked.dropna().mode()[0]
    titanic_copy['Embarked'].fillna(freq_port, inplace = True)

    titanic_copy.Fare.fillna(titanic_copy.Fare.mean(), inplace=True)

    titanic_copy.Cabin.fillna('U', inplace=True)
    titanic_copy['Cabin'] = titanic_copy['Cabin'].map(lambda c: c[0])

    return titanic_copy



def FillAge(row, grouped_train):
    condition = (
            (grouped_train['Sex'] == row['Sex']) &
            (grouped_train['Title'] == row['Title']) &
            (grouped_train['Pclass'] == row['Pclass'])
    )
    return grouped_train[condition]['Age'].values[0]


def Bin(score, breakpoints, bins):
    i = bisect.bisect(breakpoints, score)
    return bins[i]


def ToFloat(titanic_data):
    titanic_copy = titanic_data.copy()

    for col in titanic_copy:
        titanic_copy[col] = titanic_copy[col].astype(float)

    return titanic_copy


if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    main()