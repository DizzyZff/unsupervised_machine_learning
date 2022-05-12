import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
from matplotlib import cm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('wines.csv')
X = df.values
y =[]
for i in range(X.shape[0]):
    if i < X.shape[0]/2:
        y.append(0)
    else:
        y.append(1)
target_names = ['class 0', 'class 1']

mu = np.mean(X, axis=0)
C = (X - mu).T @ (X - mu)
eigvals, eigvecs = np.linalg.eig(C)

PCA = PCA(n_components=2)
PCA.fit(X)
X_PCA = PCA.transform(X)
plt.figure(figsize=(10,10))
plt.scatter(X_PCA[:,0], X_PCA[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

explained_variance = PCA.explained_variance_ratio_
print(explained_variance)
total_variance = np.sum(explained_variance)
percent_variance = total_variance*100
print(percent_variance)

#use TSNE
#perplexity 5 to 150
perplexity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
kl_divergence = []
for i in perplexity:
    tsne = TSNE(n_components=2, perplexity=i)
    X_tsne = tsne.fit_transform(X)
    kl_divergence.append(tsne.kl_divergence_)
    if i == 20:
        plt.figure(figsize=(10,10))
        plt.scatter(X_tsne[:,0], X_tsne[:,1])
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.show()
plt.figure(figsize=(10,10))
plt.plot(perplexity, kl_divergence)
plt.xlabel('Perplexity')
plt.ylabel('KL Divergence')
plt.show()

# 2-dimensional embedding using MDS
MDS = MDS(n_components=2)
X_MDS = MDS.fit_transform(X)
plt.figure(figsize=(10,10))
plt.scatter(X_MDS[:,0], X_MDS[:,1])
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()
stress = MDS.stress_/ sum(sum(MDS.dissimilarity_matrix_))
print(stress)

# use the Silhouette method to determine the optimal number of clusters
range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()
#save the plot


KMeans = KMeans(n_clusters=2, random_state=10)
cluster_labels = KMeans.fit_predict(X)
centers = KMeans.cluster_centers_
# the total sum of the distance of all points to their respective clusters centers
dis = KMeans.transform(X)
sum_dis = np.sum(dis, axis=1)
sum_dis = np.sum(sum_dis)
print(sum_dis,KMeans.inertia_)

#dBScan
from sklearn.cluster import DBSCAN
#minPoints = 2
#eps = 0.5

db = DBSCAN(eps=56, min_samples=1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(n_clusters, n_noise)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.show()




