import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
inputFeatures = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[inputFeatures]

wcssL = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X)
    wcssL.append(km.inertia_)

plt.plot(range(1, 11), wcssL)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X.loc[y_pred == i, 'Annual Income (k$)'], X.loc[y_pred == i, 'Spending Score (1-100)'], 
                s=100, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids', marker='X')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()