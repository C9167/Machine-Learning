#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# Welcome to "***Clustering (Customer Segmentation) Project***". This is the last medium project of ***Machine Learning*** course.
# 
# At the end of this project, you will have performed ***Cluster Analysis*** with an ***Unsupervised Learning*** method.
# 
# ---
# 
# In this project, customers are required to be segmented according to the purchasing history obtained from the membership cards of a big mall.
# 
# This project is less challenging than other projects. After getting to know the data set quickly, you are expected to perform ***Exploratory Data Analysis***. You should observe the distribution of customers according to different variables, also discover relationships and correlations between variables. Then you will spesify the different variables to use for cluster analysis.
# 
# The last step in customer segmentation is to group the customers into distinct clusters based on their characteristics and behaviors. One of the most common methods for clustering is ***K-Means Clustering***, which partitions the data into k clusters based on the distance to the cluster centroids. Other clustering methods include ***hierarchical clustering***, density-based clustering, and spectral clustering. Each cluster can be assigned a label that describes its main features and preferences.
# 
# - ***NOTE:*** *This project assumes that you already know the basics of coding in Python. You should also be familiar with the theory behind Cluster Analysis and scikit-learn module as well as Machine Learning before you begin.*

# ---
# ---

# # #Tasks

# Mentoring Prep. and self study####
# 
# #### 1. Import Libraries, Load Dataset, Exploring Data
# - Import Libraries
# - Load Dataset
# - Explore Data
# 
# #### 2. Exploratory Data Analysis (EDA)
# 
# 
# #### 3. Cluster Analysis
# 
# - Clustering based on Age and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Clustering based on Annual Income and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Hierarchical Clustering
# 
#     *i. Determine optimal number of clusters using Dendogram*
# 
#     *ii. Apply Agglomerative Clustering*
# 
#     *iii. Visualizing and Labeling All the Clusters*
# 
# - Conclusion

# ---
# ---

# ## 1. Import Libraries, Load Dataset, Exploring Data
# 
# There is a big mall in a specific city that keeps information of its customers who subscribe to a membership card. In the membetrship card they provide following information : gender, age and annula income. The customers use this membership card to make all the purchases in the mall, so tha mall has the purchase history of all subscribed members and according to that they compute the spending score of all customers. You have to segment these customers based on the details given.

# #### Import Libraries

# In[51]:


pip install cufflinks


# In[53]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

from ipywidgets import interact
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
import cufflinks as cf
#Enabling the offline mode for interactive plotting locally
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()

#To display the plots
get_ipython().run_line_magic('matplotlib', 'inline')
from ipywidgets import interact
import plotly.io as pio

pio.renderers.default = "notebook"


pd.set_option("display.precision", 3)
pd.options.display.float_format = '{:,.2f}'.format


# In[54]:


df = pd.read_csv("Mall_Customers.csv")


# In[55]:


df.head()


# ## 2. Exploratory Data Analysis (EDA)
# 
# After performing Cluster Analysis, you need to know the data well in order to label the observations correctly. Analyze frequency distributions of features, relationships and correlations between the independent variables and the dependent variable. It is recommended to apply data visualization techniques. Observing breakpoints helps you to internalize the data.
# 
# 
# 
# 

# In[56]:


df.info()


# In[57]:


df.shape


# In[58]:


df.rename(columns={"Annual Income (k$)" : "Annual_Income" , "Spending Score (1-100)" : "Spending_Score" } , inplace=True)


# ---
# ---

# In[59]:


df.describe()
# Outlier values can deteriorate the clustering quality.
# We should be careful about outliers in clustering algorithms.


# In[60]:


@interact(col=df.columns[1:], chart=["countplot", "histogram"])
def plot(col, chart):
    colors = np.random.choice(['blue', 'red', 'green'])
    if chart == "countplot":
        plt.figure(figsize=(20, 5))
        ax = sns.countplot(x=col, data=df)
        plt.title(col + ' Countplot')
        plt.xlabel(col)
        ax.bar_label(ax.containers[0])

    else:
        plt.figure(figsize=(20, 5))
        ax = sns.histplot(data=df, x=col, bins=80, kde=True, color=colors)
        plt.title(col + ' Histplot')
        plt.xlabel(col)
        ax.bar_label(ax.containers[0])


# We are investigating the age range of the customer group.
# We are investigating the annual revenues of customer groups.
# We are investigating the spending scores of customer groups.


# In[61]:


plt.figure(figsize=(8, 8))

explode = [0, 0.1]
plt.pie(df['Gender'].value_counts(),
        explode=explode,
        autopct='%1.1f%%',
        shadow=True,
        startangle=140)
plt.legend(labels=['Female', 'Male'])
plt.title('Male and Female Distribution')
plt.axis('off')
plt.show()
# We compare gender ratios, which is one of the important issues in customer segmentation.


# In[62]:


sns.pairplot(df);

# We can get an idea by looking at the scatterplot that "does our data tend to cluster or not?"
# If so, "how many clusters can I divide?"
# We should keep in mind the ideal clustering logic, which we specify as minimal intra cluster distance
# and maximal inter cluster distance, at every stage of clustering problems.


# In[63]:


plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.show()

# We are investigating how features are correlated to understand how they trending bivariately.


# In[64]:


plt.figure(figsize=(12, 3))
sns.swarmplot(x='Spending_Score', y='Gender', data=df, color="r")
sns.boxplot(x='Spending_Score', y='Gender', data=df, saturation=.3)
plt.title('Gender and Spending Score')
plt.show()

# stripplot : used to look at the density of categorical data.

plt.figure(figsize=(12, 6))
sns.violinplot(x='Spending_Score', y='Gender', data=df)
plt.title('Gender based Spending Score')
plt.show()

# With violinplot, we can see the density situation that we can't see in boxplot.


# In[65]:


plt.figure(figsize=(14, 8))

sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue="Gender")
plt.show()
# We look at the distribution of gender in the clustering we will create using Annual_Income and Spending_Score,
# which caught our eye above.


# In[ ]:





# # Data Processing

# In[66]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Check if 'Gender' is in the DataFrame columns
if 'Gender' in df.columns:
    # Create a LabelEncoder instance
    le = LabelEncoder()
    
    # Encode the 'Gender' column
    df['Gender'] = le.fit_transform(df['Gender'])



# In[67]:


df.head()


# ## 3. Cluster Analysis

# The purpose of the project is to perform cluster analysis using [K-Means](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) and [Hierarchical Clustering](https://medium.com/analytics-vidhya/hierarchical-clustering-d2d92835280c) algorithms.
# Using a maximum of two variables for each analysis can help to identify cluster labels more clearly.
# The K-Means algorithm requires determining the number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering), while Hierarchical Clustering builds a dendrogram without defining the number of clusters beforehand. Different labeling should be done based on the information obtained from each analysis.
# Labeling example:
# 
# - **Normal Customers**  -- An Average consumer in terms of spending and Annual Income
# - **Spender Customers** --  Annual Income is less but spending high, so can also be treated as potential target customer.

# ### Clustering based on Age and Spending Score

# #### *i. Create a new dataset with two variables of your choice*

# In[108]:


X_age_ss = df.loc[:,['Age','Spending_Score']]
X_age_ss


# In[109]:


hopkins(X_age_ss, 1)


# In[110]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)


# In[70]:


X1


# In[111]:


X1.shape


# # 

# #### *ii. Determine optimal number of clusters*

# # Modelling

# In[112]:


# function to compute hopkins's statistic for the dataframe X
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
def hopkins(X, ratio=0.05):

    if not isinstance(X, np.ndarray):
      X = X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0] * ratio) #0.05 (5%) based on paper by Lawson and Jures

    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))

    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)

    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour

    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H


# In[113]:


hopkins(X1, 1)


# In[114]:


result=[]
for _ in range(10):
    result.append(hopkins(X1, 1))
np.mean(result)


# In[115]:


import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns

# Assuming X1 is your numpy array
X1_df = pd.DataFrame(X1, columns=['Age', 'spending_Score'])
# Now, you can use X1_df for the pairplot
sns.pairplot(X1_df)


# # Clustering with K-means

# # Elbow Method
# 

# In[116]:


from sklearn.cluster import KMeans


# In[117]:


ssd = []

K = range(2,10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X1)
    ssd.append(model.inertia_)


# In[118]:


ssd


# In[119]:


pd.Series(ssd).diff()


# In[120]:


fig = px.line(x = K, y = ssd, range_x=[1,10], hover_name = pd.Series(ssd).diff().values )

fig.show()


# In[121]:


K = range(2, 10)
distortion = []
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(X1)
    distances = kmeanModel.transform(X1) # distances from each observation to each cluster centroid
    labels = kmeanModel.labels_
    result = []
    for i in range(k):
        cluster_distances = distances[labels == i, i] # distances from observations in each cluster to their own centroid
        result.append(np.mean(cluster_distances ** 2)) # calculate the mean of squared distances from observations in each cluster to their own centroid and add it to the result list
    distortion.append(sum(result)) # sum the means of all clusters and add it to the distortion list


plt.plot(K, distortion, "bo--")
plt.xlabel("Different k values")
plt.ylabel("distortion")
plt.title("elbow method")


# # Model Building and label visualisation¶

# In[122]:


import plotly.express as px


# In[123]:


model = KMeans(n_clusters=6,random_state=101)
model.fit(X1)


# In[124]:


# n_clusters = 6. Since we decided to have 6 clusters according to age and spending score.


# In[125]:


model.labels_


# In[126]:


X1 = X1.copy()
X1


# # silhoutte_score

# In[128]:


from sklearn.metrics import silhouette_score

range_n_clusters = range(2, 11)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_age_ss)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X_age_ss, cluster_labels)
    print(
        f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}"
    )


# # silhoutte score of each cluster

# In[130]:


from sklearn.cluster import KMeans

from yellowbrick.cluster import SilhouetteVisualizer

model3 = KMeans(n_clusters=6, random_state=42)
visualizer = SilhouetteVisualizer(model3)

visualizer.fit(X_age_ss)  # Fit the data to the visualizer
visualizer.poof()
plt.show()


# In[131]:


model3.n_clusters
#We can get the number of clusters with .n_clusters.


# In[132]:


#model3.n_clusters

for i in range(model3.n_clusters):
    label = (model3.labels_ == i)
    print(
        f"mean silhouette score for label {i:<4} : {visualizer.silhouette_samples_[label].mean()}"
    )
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# # iii. Apply K Means

# In[183]:


kmeans = KMeans(n_clusters=6, random_state=42)

# n_clusters = 6. Since we decided to have 4 clusters according to age and spending score.


# In[184]:


kmeans.fit_predict(X_age_ss)


# In[185]:


df_age_ss = X_age_ss.copy()
df_age_ss


# In[186]:


df_age_ss["cluster_Kmeans"] = kmeans.fit_predict(X_age_ss) #kmeans.labels_

# Add cluster_Kmeans as a column to df_age_ss.


# In[187]:


df_age_ss


# # iv. Visualizing and Labeling All the Clusters

# In[188]:


plt.figure(figsize=(14,8))
sns.scatterplot(x='Age',
                y='Spending_Score',
                hue='cluster_Kmeans',
                data=df_age_ss,
                palette="bright")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0],
            centers[:, 1],
            c='black',
            s=300,
            alpha=0.5)
plt.show()
# We look at clusters and centroids formed by age and spending score.


# # Clustering based on Annual Income and Spending Score

# # i. Create a new dataset with two variables of your choice

# In[189]:


X_ai_ss = df[['Annual_Income','Spending_Score']]
X_ai_ss.head()


# In[190]:


hopkins(X_ai_ss, 1)


# In[191]:


sns.pairplot(X_ai_ss);
# We look at the distributions of Annual_Income and Spending_Score.


# In[192]:


ssd =[]
for n in range(2,11):
    kmeans=KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_ai_ss)
    ssd.append(kmeans.inertia_) # distances from each observation to each cluster centroid
plt.figure(figsize=(10,6))
plt.plot(range(2, 11), ssd, "bo-", markersize=14.0)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('ssd')
plt.show()


# # distortion

# In[193]:


K = range(2, 10)
distortion = []
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    kmeanModel.fit(X_ai_ss)
    distances = kmeanModel.transform(X_ai_ss) # distances from each observation to each cluster centroid
    labels = kmeanModel.labels_
    result = []
    for i in range(k):
        cluster_distances = distances[labels == i, i] # distances from observations in each cluster to their own centroid
        result.append(np.mean(cluster_distances ** 2)) # calculate the mean of squared distances from observations in each cluster to their own centroid and add it to the result list
    distortion.append(sum(result)) # sum the means of all clusters and add it to the distortion list

plt.figure(figsize=(10,6))
plt.plot(K, distortion, "b*--", markersize=14.0)
plt.xlabel("Different k values")
plt.ylabel("distortion")
plt.title("elbow method")
plt.show()


# In[194]:


from sklearn.metrics import silhouette_score

range_n_clusters = range(2, 11)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_ai_ss)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X_ai_ss, cluster_labels)
    print(
        f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}"
    )


# # silhouette_score of each cluster

# In[201]:


model4 = KMeans(n_clusters=5, random_state=42)
visualizer = SilhouetteVisualizer(model4)

visualizer.fit(X_ai_ss)  # Fit the data to the visualizer
visualizer.poof()
plt.show()


# In[202]:


#model3.n_clusters

for i in range(model4.n_clusters):
    label = (model4.labels_ == i)
    print(
        f"mean silhouette score for label {i:<4} : {visualizer.silhouette_samples_[label].mean()}"
    )
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# # iii. Apply K Means

# In[203]:


kmeans2 = KMeans(n_clusters=5, random_state=42)
kmeans2.fit_predict(X_ai_ss)


# In[204]:


df_ai_ss = X_ai_ss.copy()
df_ai_ss.head()


# In[205]:


df_ai_ss['cluster_Kmeans'] = kmeans2.fit_predict(X_ai_ss) #kmeans2.labels_
df_ai_ss


# # iv. Visualizing and Labeling All the Clusters

# In[206]:


plt.figure(figsize=(15, 9))
sns.scatterplot(x='Annual_Income',
                y='Spending_Score',
                hue='cluster_Kmeans',
                data=df_ai_ss,
                palette="bright")
centers = kmeans2.cluster_centers_
plt.scatter(centers[:, 0],
            centers[:, 1],
            c='black',
            s=300,
            alpha=0.5)
plt.show()
# We look at the clusters and centroids formed according to Annual_Income and Spending_Score.


# # Hierarchical Clustering

# # i. Determine optimal number of clusters using Dendogram

# # Clustering based on Age and Spending Score¶

# # ii. Determine optimal number of clusters

# In[207]:


X_age_ss


# In[208]:


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram


# # Dendrogram

# In[209]:


@interact(method=["ward", "complete", "average", "single"])
def dendogramer(method):
    link = linkage(y=X_age_ss, method=method)
    plt.figure(figsize=(20, 10))
    plt.title("Dendogram")
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    dendrogram(link,
               truncate_mode="lastp",
               p=10,
               show_contracted=True,
               leaf_font_size=10)


# In[210]:


range_n_clusters = range(2, 11)
for num_clusters in range_n_clusters:
    # intialise kmeans
    Agg_model = AgglomerativeClustering(n_clusters=num_clusters)
    Agg_model.fit(X_age_ss)
    cluster_labels = Agg_model.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X_age_ss, cluster_labels)
    print(
        f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}"
    )


# # Clustering based on Annual Income and Spending Score

# In[211]:


X_ai_ss


# In[212]:


@interact(method=["ward", "complete", "average", "single"])
def dendogramer(method):
    link = linkage(y=X_ai_ss, method=method)
    plt.figure(figsize=(20, 10))
    plt.title("Dendogram")
    plt.xlabel("Observations")
    plt.ylabel("Distance")
    dendrogram(link,
               truncate_mode="lastp",
               p=10,
               show_contracted=True,
               leaf_font_size=10)


# In[213]:


range_n_clusters = range(2, 11)
for num_clusters in range_n_clusters:
    # intialise kmeans
    Agg_model = AgglomerativeClustering(n_clusters=num_clusters)
    Agg_model.fit(X_ai_ss)
    cluster_labels = Agg_model.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X_ai_ss, cluster_labels)
    print(
        f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}"
    )


# # ii. Apply Agglomerative Clustering¶

# In[214]:


X_age_ss


# In[215]:


Agg1 = AgglomerativeClustering(
    n_clusters=4,
    metric=
    'euclidean',  # If linkage = "ward" then metric='euclidean' is required.
    linkage='ward')  # originating from the formulation of variance...
y_agg = Agg1.fit_predict(X_age_ss)


# In[216]:


df_age_ss


# In[217]:


df_age_ss['cluster_Agg'] = y_agg
df_age_ss.head()


# # Annual Income and Spending Score

# In[218]:


X_ai_ss


# In[219]:


Agg2 = AgglomerativeClustering(n_clusters=5,
                               metric='euclidean',
                               linkage='ward')
y_agg2 = Agg2.fit_predict(X_ai_ss)


# In[220]:


df_ai_ss


# In[221]:


df_ai_ss['cluster_Agg'] = y_agg2
df_ai_ss.head()


# # iii. Visualizing and Labeling All the Clusters

# Age and Spending Score

# In[222]:


df_age_ss


# In[223]:


plt.figure(figsize=(14, 9))
sns.scatterplot(x='Age',
                y='Spending_Score',
                hue='cluster_Agg',
                data=df_age_ss,
                palette="viridis")
plt.show()


# In[ ]:





# Annual Income and Spending Score¶

# In[224]:


df_ai_ss


# In[225]:


plt.figure(figsize=(14,9))
sns.scatterplot(x='Annual_Income',
                y='Spending_Score',
                hue='cluster_Agg',
                data=df_ai_ss ,
                palette="bright")
plt.show()


# In[226]:


plt.figure(figsize=(20, 8))

plt.subplot(121)
sns.scatterplot(x='Annual_Income',
                y='Spending_Score',
                hue='cluster_Kmeans',
                data=df_ai_ss,
                palette=['green', 'orange', 'brown', 'dodgerblue', 'red'])
plt.title("K_means")
plt.subplot(122)
sns.scatterplot(x='Annual_Income',
                y='Spending_Score',
                hue='cluster_Agg',
                data=df_ai_ss,
                palette=['orange', 'green', 'red', 'dodgerblue', 'brown'])
plt.title("Agg")
plt.show()


# # Interpretation based on Age and Spending Score

# In[227]:


df_age_ss


# In[228]:


# lets see the number of poeple lie in each group
plt.title("clusters with the number of customers")
plt.xlabel("clusters")
plt.ylabel("Count")
ax = df_age_ss.cluster_Kmeans.value_counts().plot(kind='bar')
ax.bar_label(ax.containers[0])
plt.show()


# In[229]:


df["cluster_Age_Spending_Score"] = df_age_ss.cluster_Kmeans
df.head()

# We add clusters resulting from Kmeans to our df (age-spending score)


# In[230]:


plt.title("Men VS Women ratio in each cluster")
plt.ylabel("Count")
ax = sns.countplot(x=df.cluster_Age_Spending_Score, hue=df.Gender)
for p in ax.containers:
    ax.bar_label(p)
plt.show()
# We count clusters according to gender.


# In[231]:


df.groupby("cluster_Age_Spending_Score").mean()
# We group them according to the clusters formed.


# In[232]:


plt.figure(figsize=(20, 6))

plt.subplot(131)
sns.boxplot(y="Age", x="cluster_Age_Spending_Score", data=df)
sns.swarmplot(y="Age",
              x="cluster_Age_Spending_Score",
              data=df,
              color="blue")

plt.subplot(132)
sns.boxplot(y="Annual_Income", x="cluster_Age_Spending_Score", data=df)
sns.swarmplot(y="Annual_Income",
              x="cluster_Age_Spending_Score",
              data=df,
              color="blue")

plt.subplot(133)
sns.boxplot(y="Spending_Score", x="cluster_Age_Spending_Score", data=df)
sns.swarmplot(y="Spending_Score",
              x="cluster_Age_Spending_Score",
              data=df,
              color="blue")
plt.show()


# In[233]:


ax = df.groupby("cluster_Age_Spending_Score").mean().plot(kind='bar',
                                                          figsize=(20, 6),
                                                          fontsize=20)
for p in ax.containers:
    ax.bar_label(p, fmt="%.0f", size=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Conclusion

# **cluster 0** : The average age is around 30, both annula_income 66 spending_scores 82 are on average.
# It should be researched what can be done to direct to more spending.
# Attention Customers
# 
# **cluster 1**: The average age is around 51, the annula_income is high but the spending_scores are very low.
# This group is our target audience and specific strategies should be developed to drive this group to spend.
# Lost Customers
# 
# **cluster 2** :The average age is around 46. The annula_income is high and spending_scores are very high.
# This group consists of our loyal customers. Our company derives the main profit from this group. Very
# special promotions can be made in order not to miss it.
# Loyal Customers
#     
# **cluster 3**: The average age is around 25.both annula_income and spending_scores are on average.
# It should be researched what can be done to direct to more spending.
# loyal Candidate Customers
# 
# **cluster 4**:The average age is around 64.both annula_income and spending_scores are on high.
# Age Attention Customers
#     
# **cluster 5**:The average age is around 31. annual_income high but spending_score is too low. Risky group.
# Risky Customers

# # Recomendations

# 
# 1. **Recognition and Appreciation:** Show your appreciation for their loyalty. Send thank-you notes, offer special discounts or gifts, and recognize their loyalty on social media or through your company's loyalty program.
# 
# 2. **Personalization:** Use the data you have on their preferences and buying behavior to offer personalized recommendations and discounts. Make them feel special by catering to their specific needs and tastes.
# 
# 3. **Exclusive Offers:** Offer exclusive deals and promotions to your loyal customers. This not only shows your appreciation but also gives them an incentive to continue shopping with your brand.
# 
# 4. **Feedback and Engagement:** Ask for their feedback and opinions. They can provide valuable insights into your products or services. Engaging them in surveys or discussions makes them feel valued.
# 
# 5. **Communication:** Keep them informed about new products, services, or updates. Regular newsletters or updates can keep your brand in their minds and encourage repeat purchases.
# 
# 6. **Loyalty Programs:** If you don't already have one, consider starting a loyalty program. Offer rewards, points, or tiers that provide increasing benefits for long-term customers.
# 
# 7. **Customer Support:** Provide excellent customer service. Address their issues promptly, offer easy returns, and ensure that their shopping experience is hassle-free.
# 
# 8. **Surprise and Delight:** Occasionally surprise them with unexpected gifts or offers. These "wow" moments can be memorable and strengthen their loyalty.
# 
# 9. **Community Building:** Consider building a community around your brand where loyal customers can connect with each other. It fosters a sense of belonging and can create a stronger bond with your brand.
# 
# 10. **Consistency:** Maintain the quality and standards that they initially loved about your brand. Consistency in product or service quality is key to retaining loyal customers.
# 
# 11. **Ask for Referrals:** Loyal customers are more likely to refer friends and family. Encourage them to refer others by offering referral incentives.
# 
# 12. **Social Proof:** Share positive reviews and testimonials from your loyal customers on your website and social media. This can build trust and attract more loyal customers.
# 
# 

# In[234]:


df_ai_ss


# In[235]:


df_ai_ss.cluster_Kmeans.value_counts()


# In[236]:


# lets see the number of poeple lie in each group
plt.title("clusters with the number of customers")
plt.xlabel("clusters")
plt.ylabel("Count")
ax = df_ai_ss.cluster_Kmeans.value_counts().plot(kind='bar')
ax.bar_label(ax.containers[0]);


# In[237]:


df.head()


# In[238]:


df.drop(columns="cluster_Age_Spending_Score", inplace=True)


# In[242]:


df["cluster_Annual_Income_Spending_Score"] = df_ai_ss.cluster_Kmeans
df.head()
# Add cluster_Annual_Income_Spending_Score column.


# In[243]:


df.head()


# In[244]:


plt.title("Men VS Women ratio in each cluster")
plt.ylabel("Count")
ax =sns.countplot(x=df.cluster_Annual_Income_Spending_Score, hue=df.Gender)
for p in ax.containers:
    ax.bar_label(p)


# In[245]:


df.groupby(["Gender", "cluster_Annual_Income_Spending_Score"]).mean()

# here we group df by both Gender and cluster_Annual_Income_Spending_Score.


# In[246]:


plt.figure(figsize = (20,6))

plt.subplot(131)
sns.boxplot(y="Age", x="cluster_Annual_Income_Spending_Score",
            hue= "Gender", data = df,palette="deep",saturation=0.5)
sns.swarmplot(y = "Age", x = "cluster_Annual_Income_Spending_Score",
              hue= "Gender", data = df,palette=sns.color_palette("Paired"))

plt.subplot(132)
sns.boxplot(y="Annual_Income", x="cluster_Annual_Income_Spending_Score",
            hue="Gender", data = df, palette="deep",saturation=0.5)
sns.swarmplot(y = "Annual_Income", x = "cluster_Annual_Income_Spending_Score",
              hue= "Gender", data = df,palette=sns.color_palette("Paired"))

plt.subplot(133)
sns.boxplot(y="Spending_Score", x="cluster_Annual_Income_Spending_Score",
            hue="Gender", data=df, palette="deep",saturation=0.5);
sns.swarmplot(y = "Spending_Score", x = "cluster_Annual_Income_Spending_Score",
              hue= "Gender", data = df,palette=sns.color_palette("Paired"))
plt.show()


# ## Conclusion

# In[ ]:





# In[249]:


ax = df.groupby(["Gender", "cluster_Annual_Income_Spending_Score"]).mean().plot(kind="bar",
                                                                                figsize=(20,6),
                                                                                fontsize=20)
for p in ax.containers:
    ax.bar_label(p, fmt="%.0f", size=14)


# ### Female
# 
# **cluster 0** : The average age is around 40, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 30, the annula_income is very high and the spending_scores is high.
# This group is our target audience and special strategies need to be developed for this group. It can be
# directed to shopping with gift certificates.    
# 
# **cluster 2** :The average age is around 23. Both annula_income high but spending_score is low.Loyal Customers
# 
# **cluster 3**: The average age is around 43. Very High annual_incomes but very low spending scores.
# Attention Customers
# 
# **cluster 4**: The average age is around 43, the annual income and the spending_score
# is very low.
# 
# 

# ### Male
# 
# **cluster 0** : The average age is around 45, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 33, the annula_income is very high and the spending_scores is also high.
# This group is our target audience and special strategies need to be developed for this group.    
# 
# **cluster 2** :The average age is around 23. Both annula_income islow but spending score is too high.This
# group does a lot of shopping, but they do not bring much profit Loyal Customers.
# 
# **cluster 3**: The average age is around 39. High annual_incomes but very low spending scores.attention Customers
# 
# **cluster 4**: The average age is around 48, the annual income and the spending_score
# is very low.

# In[251]:


ax = df.groupby("cluster_Annual_Income_Spending_Score").mean().plot(kind='bar', figsize = (20,6))
for p in ax.containers:
    ax.bar_label(p, fmt="%.0f")


# **cluster 0** : The average age is around 43, both annula_income and spending_scores are on average.
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 33, both annula_income and spending_scores are very high.
# This group consists of our loyal customers. Our company derives the main profit from this group. Very
# special promotions can be made in order not to miss it.
# 
# **cluster 2** :The average age is around 25.annula_income is low but spending_score is high.Loyal Customers
# 
# **cluster 3**: The average age is around 41. High annual_incomes but very low spending scores.atteintion Customers
# 
# **cluster 4**: The average age is around 45, their annual income is very low and their spending_score
# is very low. Loser Customers.

# 
# 

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
