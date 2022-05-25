import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import *
from matplotlib.pyplot import *
# from Pre_processing import *

import Filter as ft


class Model:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.x = None
        self.y = None

    def addYColumn(self):
        self.y = abs(self.data['revenue_adj'] - self.data['budget_adj'])
        self.data['net_profit'] = self.y

    def addXColumn(self):
        self.x = self.data.iloc[:, :]
        self.x = self.x.drop(['revenue_adj', 'budget_adj', 'net_profit'], axis=1)
        print(len(list(self.x)), list(self.x))

    def linearRegression(self):

        self.addYColumn()
        self.addXColumn()

        # Split the data to training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25)

        self.model = LinearRegression()
        self.model.fit(self.x, self.y)

    def polynomialRegression(self):

        self.addYColumn()
        self.addXColumn()

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, shuffle=False)

        model = PolynomialFeatures(degree=5)
        X = model.fit_transform(X_train)

        poly_model = linear_model.LinearRegression()
        poly_model.fit(X, y_train)

        score = poly_model.score(model.fit_transform(X_train), y_train)
        coef = poly_model.coef_
        intercept = poly_model.intercept_
        prediction = poly_model.predict(model.fit_transform(X_test))

        print("score", score)
        print("coef", coef)
        print("intercept", intercept)
        print("prediction", prediction)

    def fitMethod(self):
        self.linearRegression()
        # self.polynomialRegression()

    def getCoefficient(self):
        self.fitMethod()
        return self.model.coef_

    def getScore(self):
        self.fitMethod()
        return self.model.score(self.x, self.y)

    def getIntercept(self):
        self.fitMethod()
        return self.model.intercept_

    def getPrediction(self, newData):
        self.fitMethod()
        return self.model.predict(newData)
