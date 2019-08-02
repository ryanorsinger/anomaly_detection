# Detect Anomalies using Density Based Clustering
# Clustering-Based Anomaly Detection

# Assumption: Data points that are similar tend to belong to similar groups or clusters, as determined by their distance from local centroids. Normal data points occur around a dense neighborhood and abnormalities are far away.
# Using density based clustering, like DBSCAN, we can design the model such that the data points that do not fall into a cluster are the anomalies.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./customers.csv")

df = df[["Grocery", "Milk", "Fresh"]]

columns = list(df.columns)
columns

np_array = df.values.astype("float32", copy = False)

scaler = MinMaxScaler().fit(np_array)

np_array = scaler.transform(np_array)

dbsc = DBSCAN(eps = .10, min_samples = 20).fit(np_array)

labels = dbsc.labels_

scaled_columns = ["Scaled_" + column for column in columns]
scaled_columns

# Create a dataframe containing the scaled valuese
scaled_df = pd.DataFrame(np_array, columns=scaled_columns)

df['labels'] = labels
df.labels.value_counts()

# Merge the scaled and non-scaled values into one dataframe
df = df.merge(scaled_df, on=df.index)
df = df.drop(columns=['key_0'])
df.head()

## Break out the outliers vs. the inliers vs. the population values
df.describe()
df[df.labels == 0].describe()
df[df.labels == -1].describe()

sns.scatterplot(df["Scaled_Grocery"], df["Scaled_Milk"], hue=df.labels)
plt.show()

sns.scatterplot(df.Grocery, df.Milk, hue=df.labels)
plt.show()


sns.scatterplot(df.Grocery, df.Fresh, hue=df.labels)
plt.show()

sns.scatterplot(df.Milk, df.Fresh, hue=df.labels)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# plot the points
ax.scatter(df.Fresh, df.Milk, df.Grocery, c=df.labels, edgecolor='k')


ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('Fresh')
ax.set_ylabel('Milk')
ax.set_zlabel('Grocery')    