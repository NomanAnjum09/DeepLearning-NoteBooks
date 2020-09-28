import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import scatter_matrix
data = pd.read_csv('sonar.all-data',header=None)
print(data.shape)
#print(data.head())

X = data[data.columns[0:60]]
Y = data[data.columns[60]]

#data.hist()
#plt.show()

scatter_matrix(data)
plt.show()