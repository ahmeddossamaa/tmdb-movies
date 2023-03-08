import pandas as pd
import numpy as np
from numpy import array

import Filter as ft
import Model as md

data = pd.read_csv('dataset/tmdb-movies.csv')
data = pd.DataFrame(data)
data = ft.Filter(data).getFilteredFile()

coefficient = md.Model(data).polynomialRegression()
