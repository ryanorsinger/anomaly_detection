from __future__ import division

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy import linspace, loadtxt, ones, convolve
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import collections
import math
from sklearn import metrics
from random import randint
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from random import randint
%matplotlib inline
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def prepare(df):
    df = df[["resource", "user", "ip_address"]]

    # Set missing user to "Unknown"
    df.user.fillna('Unknown', inplace=True)

    # Encode discrete variables
    encoder = LabelEncoder()

    df.resource.value_counts()

    encoder.fit(df.resource)

    df["encoded_resource"] = encoder.transform(df.resource)

    # Set this aside for decoding, yo!
    resource_df = df[["resource", "encoded_resource"]]

    encoder.fit(df.user)
    df["encoded_user"] = encoder.transform(df.user)
    user_df = df[["user", "encoded_user"]]


    encoder.fit(df.ip_address)
    df["encoded_ip_address"] = encoder.transform(df.ip_address)
    ip_address_df = df[["ip_address", "encoded_ip_address"]]

    encoded_df = df[["encoded_resource", "encoded_user", "encoded_ip_address"]] 
    df = df.drop(columns = ["encoded_resource", "encoded_user", "encoded_ip_address"])
    df.head()
    encoded_df.head()

    return (df, encoded_df)

colnames=['timestamp', 'resource', 'user', 'cohort_id', 'ip_address']

df = pd.read_csv("./data/curriculum-access.txt", sep='\t', header=None,
                 index_col=False,
                 names=colnames)

# Setup the original and encoded dataframes
df, encoded_df = prepare(df)



# np_array = encoded_df.values.astype("float32", copy = False)

# stscaler = StandardScaler().fit(np_array)
# np_array = stscaler.transform(np_array)

# dbsc = DBSCAN(eps = .75, min_samples = 15).fit(np_array)
# labels = dbsc.labels_
# labels[0:10]

df.head()

# df.groupby("resource").agg(['min','max'])
# df.ip_address.groupby(df.user).count()

# x = df[df.user == "Justin Reich"].groupby(["user", "ip_address", "resource"]).agg("count")


df = df.reset_index()
df.groupby(["user", "ip_address", "resource"]).agg("count")

df.groupby(["resource", "ip_address"]).count()

df.groupby(["resource", "user"]).count()

grouped = df.groupby(["user", "ip_address", "resource"]).count()
grouped

users = grouped.groupby(level=0).count()
users

ip_addresses = grouped.groupby(level=1).count()
ip_addresses

resources = grouped.groupby(level=2).count()
resources

number_of_pages = len(resources)
number_of_pages
