# Density-Based Anomaly Detection 

Density-based anomaly detection is based on the k-nearest neighbors algorithm.
Assumption: Normal data points occur around a dense neighborhood and abnormalities are far away. 


K-nearest neighbor: k-NN is a simple, non-parametric lazy learning technique used to classify data based on similarities in distance metrics such as Eucledian, Manhattan, Minkowski, or Hamming distance.
Relative density of data: This is better known as local outlier factor (LOF). This concept is based on a distance metric called reachability distance.


## Clustering-Based Anomaly Detection 
Clustering is one of the most popular concepts in the domain of unsupervised learning.
Assumption: Data points that are similar tend to belong to similar groups or clusters, as determined by their distance from local centroids.
K-means is a widely used clustering algorithm. It creates 'k' similar clusters of data points. Data instances that fall outside of these groups could potentially be marked as anomalies.


Depending on the use case, the output of an anomaly detector could be numeric scalar values for filtering on domain-specific thresholds or textual labels (such as binary/multi labels).
