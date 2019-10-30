import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Загрузка выборки из файла titanic.csv с помощью пакета Pandas.
data = pd.read_csv('titanic.csv', index_col='PassengerId')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

#Удаление их из выборки, у которых есть пропущенные признаки.
data = data.dropna()

#Остаются четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
#Признак Sex имеет строковые значения.
#Целевая переменная — столбец Survived.
X = data[['Pclass', 'Fare', 'Age', 'Sex']]
X = X.replace({'male': 0, 'female': 1})
y = data['Survived']

#Обучение решающего дерева с параметром random_state=241 и остальными параметрами по умолчанию
clf = DecisionTreeClassifier(random_state=241)
clf = clf.fit(X, y)

#Вычисление важности признаков
importances = clf.feature_importances_
print(importances)