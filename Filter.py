import pandas as pd
import numpy as np
from sklearn import preprocessing


class Filter:

    def __init__(self, file):
        self.data = file
        self.x = None
        self.y = None
        self.cast = None
        self.genres = None

    # Converters
    def convertZerosToMeanValue(self):
        df = self.data[self.data['budget_adj'] != 0]
        self.data['budget_adj'].replace(0, np.mean(df['budget_adj']), inplace=True)

        df = self.data[self.data['revenue_adj'] != 0]
        self.data['revenue_adj'].replace(0, np.mean(df['revenue_adj']), inplace=True)

    # Columns
    def setUpColumns(self):
        columnsToBeDropped = {'id', 'imdb_id', 'original_title', 'homepage', 'keywords', 'overview', 'tagline', 'director'}
        self.convertZerosToMeanValue()
        self.data.drop(columnsToBeDropped, axis=1, inplace=True)

    def removeDuplicatesAndNull(self):
        # remove duplicates
        self.data.drop_duplicates(inplace=True)
        # remove null
        self.data.dropna(inplace=True)

    def convertCategoricalToNumerical(self):
        label_encoder = preprocessing.LabelEncoder()
        for i in self.data:
            if self.data[i].dtype.name == "object":
                self.data[i] = label_encoder.fit_transform(self.data[i])

    # Getters
    def getFilteredFile(self):
        self.setUpColumns()
        self.removeDuplicatesAndNull()
        self.convertCategoricalToNumerical()
        return self.data
