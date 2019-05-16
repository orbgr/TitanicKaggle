# data analysis and wrangling
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# dependencies
from Utils.utils import *


split = 10
count = acc = recall = precision = max_acc = 0
combine = pd.DataFrame()
prob_cabin = []
name_to_title = {
    "Capt": "Job",
    "Col": "Job",
    "Major": "Job",
    "Jonkheer": "Job",
    "Don": "Job",
    "Sir": "Mr",
    "Dr": "Job",
    "Rev": "Job",
    "the Countess": "Mrs",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Job",
    "Lady": "Miss"
}


def main():
    # input
    get_combined_data()

    # data preparation
    Cleansing()
    Blending()
    train, test, targets = RecoverTrainTarget()
    # classifies
    train_reduced, test_reduced = FeatureSelections(train,targets,test)
    print(f'Data Sample After Reduced:\n {combine.head(10)}')
    best_model = ModelsPerformance(train_reduced, targets)

    KaggleTest(train_reduced,targets,test_reduced,best_model)


def get_combined_data():
    global combine
    train = pd.read_csv('./DataTitanic.csv')
    test = pd.read_csv('./test.csv')

    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)
    combine = train.append(test)
    combine.reset_index(inplace=True)
    combine.drop(['index', 'PassengerId'], inplace=True, axis=1)


def RecoverTrainTarget():
        global combine

        targets = pd.read_csv('./DataTitanic.csv', usecols=['Survived'])['Survived'].values
        train = combine.iloc[:891]
        test = combine.iloc[891:]

        return train, test, targets


def Blending():
    global combine
    global name_to_title
    global prob_cabin


    # +CabinDigit extract \d from tickets
    combine['CabinDigit'] = pd.to_numeric(combine['Cabin'].replace('[\D]+', '', regex=True)).fillna(0).astype(int)

    ImputeMissingCabin()

    # +FamilySize   -SibSp  -Parch
    combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1
    combine.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # +Title as number  -Name
    pattern = r',\s+(\w[\w\s]+)\.'
    combine['Name'] = combine['Name'].str.extract(pattern, expand=True)

    combine['Name'] = combine['Name'].map(name_to_title).fillna("Mr")
    combine.rename(columns={'Name': 'Title'}, inplace=True)

    # Filling missing Age
    grouped_train = combine.groupby(['Sex', 'Pclass', 'Title']).median().reset_index()[
        ['Sex', 'Pclass', 'Title', 'Age']]  # pivoting on Sex,Class,Title to median Age
    combine['Age'] = combine.iloc[:].apply(
        lambda row: FillAge(row, grouped_train) if np.isnan(row['Age']) else row['Age'], axis=1)

    # +TicketAlphaBet extract \D from tickets
    combine['TicketAlphaBet'] = combine['Ticket'].replace('[\d\./]+', '', regex=True)
    combine['TicketAlphaBet'] = combine['TicketAlphaBet'].apply(
        lambda row: row[0] if len(row) > 0 else '?')

    # +TicketAlphaBet extract \d from tickets
    combine['TicketDigit'] = pd.to_numeric(combine['Ticket'].replace('[\D]+', '', regex=True)).fillna(0).astype(int)


    # +TicketAmount a unique tickets as number -Ticket
    combine['Ticket'] = combine['Ticket'].replace('\D', '', regex=True)
    combine['Ticket'] = combine.groupby('Ticket')['Ticket'].transform('count')
    combine['Ticket'] = [Bin(ticket, breakpoints=[1, 3, 5, 100], bins='1234')
                         for ticket in combine['Ticket']]
    combine.rename(columns={'Ticket': 'TicketAmount'}, inplace=True)

    # binning
    combine['CabinDigit'] = [
        Bin(size, breakpoints=[1, 50, 100, float("inf")], bins=['?', 'LeftCabin', 'CenterCabin', 'RightCabin'])
        for size in combine['CabinDigit']]

    combine['FareGroup'] = [Bin(size, breakpoints=[8, 15, 31, float("inf")], bins='12345')
                                  for size in combine['Fare']]
    combine['AgeGroup'] = [Bin(age, breakpoints=[5,10, 20, 50, float("inf")], bins=['Baby','Child','Youth','Adult','Old'])
                                  for age in combine['Age']]
    combine['FamilyGroupSize'] = [Bin(size, breakpoints=[1, 3, 5, float("inf")], bins=['Alone','SmallFam','MedFam','BigFam','HugeFam'])
                                  for size in combine['FamilySize']]

    # encoding in dummy variable
    # combine = AddDummies(combine, ['TicketAlphaBet','FamilyGroupSize','FareGroup','CabinDigit','AgeGroup'],['TicketAlphaBet', 'FamilyGroupSize','AgeGroup','FareGroup','CabinDigit'])
    CategoryToNumeric(combine,["Sex","Embarked","Cabin","Title","TicketAlphaBet","FamilyGroupSize","FareGroup","CabinDigit","AgeGroup"])

    # prob_cabin = combine.loc[:891,"Cabin"].value_counts()
    # prob_cabin = prob_cabin.div(prob_cabin.sum())
    # prob_cabin = prob_cabin.to_dict()


def Cleansing():
    global combine
    freq_port = combine.Embarked.dropna().mode()[0]
    combine.loc[:891,"Embarked"].fillna(freq_port, inplace=True)
    # combine.loc[:,"Cabin"].fillna("U", inplace=

    # todo: Ask about fill NA for test
    combine.loc[:, "Fare"].fillna(combine.Fare.mean(), inplace=True)
    combine.loc[:, "Embarked"].fillna(freq_port, inplace=True)


def ImputeMissingCabin():
    global combine

    combine_filled = combine
    combine_filled['Cabin'] = combine_filled.iloc[:].apply(
        lambda row: row['Cabin'] if pd.isnull(row['Cabin']) else ord(row['Cabin'][0]), axis=1)
    combine_filled = combine_filled.drop(["Name", "Sex", "Ticket", "Embarked"], axis=1)
    combine['Cabin'] = ImputeMissing(combine_filled)[5]
    combine['Cabin'] = combine.iloc[:].apply(
        lambda row: row['Cabin'] if pd.isnull(row['Cabin']) else chr(int(row['Cabin'])), axis=1)
    combine['Cabin'] = combine.loc[:, "Cabin"].map(lambda c: c[0])


def FillAge(row, grouped_train):
    condition = (
            (grouped_train['Sex'] == row['Sex']) &
            (grouped_train['Title'] == row['Title']) &
            (grouped_train['Pclass'] == row['Pclass'])
    )
    return grouped_train[condition]['Age'].values[0]


if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    main()