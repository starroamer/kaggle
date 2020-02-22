import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


def main():
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")

    # Ticket字段重复值过多，cabin字段缺失值太多，去掉
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    # 从Name中抽取Title特征
    for dataset in combine:
        dataset["Title"] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # print(pd.crosstab(train_df['Title'], train_df['Sex']))
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # 有了Title特征后，去掉Name字段
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    # Sex字段转成整数值
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # 补充缺失的Age字段
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_mean = guess_df.median()
                age_std = guess_df.std()
                age_guess = random.uniform(age_mean - age_std, age_mean + age_std)

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = \
                    guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    # Age分桶
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['SibSp'] + dataset['Parch'] == 0, 'IsAlone'] = 1

    train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    print(train_df.head())
    np.concatenate


if __name__ == "__main__":
    main()
