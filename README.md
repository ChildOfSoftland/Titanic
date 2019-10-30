# Titanic: Importance of symptoms

## Introduction
Decision trees belong to the class of logical methods. Their main idea is to combine a certain number of simple decision rules, so that the resulting algorithm is interpretable. As the name implies, the decision tree is a binary tree in which each vertex is associated with a certain rule of the form "the jth sign has a value less than b". The leaves of this tree are written prediction numbers. To get the answer, you need to start from the root and make transitions to either the left or the right subtree, depending on whether the rule from the current vertex is fulfilled or not.

One of the features of decisive trees is that they allow you to obtain the importance of all the traits used. The importance of a trait can be estimated based on how much the quality criterion has improved due to the use of this trait in the tree tops.

### Data
In this assignment, we will again examine the data on the passengers of the Titanic. We will solve the classification problem on them, in which, according to various characteristics of passengers, it is required to predict which of them survived after the ship wreck.

### Scikit-Learn implementation
In the scikit-learn library, decision trees are implemented in the sklearn.tree.DecisionTreeСlassifier (for classification) and sklearn.tree.DecisionTreeRegressor (for regression) classes. Model training is done using the fit function.

### Usage example:
```
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
```

In this quest, you will also need to find the importance of the symptoms. This can be done with an already trained classifier:
```
importances = clf.feature_importances_
```

The 'importances' variable will contain an array of "importance" attributes. The index in this array corresponds to the attribute index in the data.

It is worth noting that the data may contain omissions. Pandas stores values ​​like 'nan' (not a number). In order to check if the number is nan, you can use the 'np.isnan' function.

## Prerequisits
- [Python 3.7](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)

## Installation
```
pip install pandas
pip install numpy
pip install scikit-learn

```
