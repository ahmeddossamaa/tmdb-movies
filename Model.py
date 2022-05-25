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


class Model:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.x = None
        self.y = None

    def yColumn(self):
        self.y = self.data['revenue_adj'] - self.data['budget_adj']
        self.data['net_profit'] = self.y

    def xColumn(self):
        self.x = self.data.iloc[:, :]
        self.x = self.x.drop(['revenue_adj', 'budget_adj', 'net_profit'], axis=1)

    def polynomialRegression(self):
        self.yColumn()
        self.xColumn()

        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, shuffle=False)

        model = PolynomialFeatures(degree=5)
        X = model.fit_transform(X_train)
        X_test_poly = model.fit_transform(X_test)

        poly_model = linear_model.LinearRegression()
        poly_model.fit(X, y_train)

        coef = poly_model.coef_
        intercept = poly_model.intercept_
        prediction = poly_model.predict(X_test_poly)
        score = poly_model.score(X, y_train)

        # print("coef", coef)
        # print("intercept", intercept)
        print("prediction", prediction)
        print("score", score)
