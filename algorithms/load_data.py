import csv
import numpy as np
import pandas as pd

# Download data from https://archive.ics.uci.edu/ml/datasets/spambase
FILE_NAME = "spambase.data"

# 1) load with csv file
with open(FILE_NAME, "r") as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, dtype=np.float32)
print(data.shape)

# 2) load with np.loadtxt()
# skiprows=1
data = np.loadtxt(FILE_NAME, delimiter=",", dtype=np.float32)
print(data.shape, data.dtype)

# 3) load with np.genfromtxt()
# skip_header=0, missing_values="---", filling_values=0.0
data = np.genfromtxt(FILE_NAME, delimiter=",", dtype=np.float32)
print(data.shape)

# split into X and y
n_samples, n_features = data.shape
n_features -= 1

X = data[:, 0:n_features]
y = data[:, n_features]

print(X.shape, y.shape)
print(X[0, 0:5])
# or if y is the first column
# X = data[:, 1:n_features+1]
# y = data[:, 0]

# 4) load with pandas: read_csv()
# na_values = ['---']
df = pd.read_csv(FILE_NAME, header=None, skiprows=0, dtype=np.float32)
df = df.fillna(0.0)

# dataframe to numpy
data = df.to_numpy()
print(data[4, 0:5])

# convert datatypes in numpy
# data = np.asarray(data, dtype = np.float32)
# print(data.dtype)
