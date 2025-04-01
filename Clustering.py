
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics import silhouette_samples,silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import decomposition
import plotly.graph_objects as go

veriler =pd.read_csv('drug200.csv')

veriler.info()

def label_encoder(y):
     le = LabelEncoder()
     veriler[y] = le.fit_transform(veriler[y])


label_list = ["Sex","BP","Cholesterol","Na_to_K","Drug"]

for l in label_list:
    label_encoder(l)

X, y = veriler.drop(['Drug'], axis=1), veriler['Drug']
X = veriler.iloc[:,5:].values
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_frame()
y_test = y_test.to_frame()

one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train).todense()
y_test_hot =  one_hot.transform(y_test).todense()

def scale_a_column(data, label_column='Drug'):
    scaler = StandardScaler()
    df1 = data.drop([label_column], axis=1)
    res = scaler.fit_transform(df1)
    df1 = pd.DataFrame(res, columns=df1.columns)
    df1[label_column] = data[label_column]
    return df1

scaled_data = scale_a_column(veriler)

plt.figure(figsize = (9,5))
sns.swarmplot(x = "Drug", y = "Na_to_K", data = veriler)
plt.legend(veriler.Drug.value_counts().index)
plt.title("Na_to_K -- Drug")
plt.show()

def plot_2d(data, x, y, figsize=(9, 5)):
    plt.figure(figsize = figsize)
    sns.swarmplot(x = x, y = y, hue="Drug",data=data)
    plt.legend()
    plt.title(f"{x} -- {y} -- Drug")
    plt.show()

plot_2d(scaled_data, 'Sex', 'Na_to_K')

plot_2d(scaled_data, 'BP', 'Na_to_K')

plot_2d(scaled_data, 'Cholesterol', 'Na_to_K')

plot_2d(scaled_data, 'Cholesterol', 'BP')

X_scaled, y_scaled = scaled_data.drop(['Drug'], axis=1), scaled_data['Drug']

def run_kmean(X_scaled, y, n_clusters=14):
    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42)

    X_kmeans = X_scaled.copy()
    #X_kmeans['Na_to_K'] = X_kmeans['Na_to_K'] * 10
    #X_kmeans['Age'] = X_kmeans['Age'] * 5
    #X_kmeans['BP'] = X_kmeans['BP'] * 100
    #X_kmeans['Cholesterol'] = X_kmeans['Cholesterol'] * 1000

    kmeans.fit(X_kmeans)

    X_kmeans = X_scaled.copy()
    X_kmeans['cluster'] = kmeans.labels_
    print(X_kmeans)
    for i in range(0, X_kmeans.shape[1]):
        for j in range(i+1, X_kmeans.shape[1]):
            for k in range(j+1, X_kmeans.shape[1]-1):
                break
                plot_one(X_kmeans, X_kmeans.columns[i], X_kmeans.columns[j], X_kmeans.columns[k], 'cluster')
    #plot_one(X_kmeans, 'BP', 'Na_to_K', 'Cholesterol', 'cluster')
    return X_kmeans


def run_em(X_scaled, y):
    X_GM = X_scaled.copy()
    gm = GaussianMixture(n_components=14, random_state=0).fit(X_GM)
    labels = gm.predict(X_GM)

    X_GM = X_scaled.copy()
    X_GM['cluster'] = labels
    print(X_GM)

from sklearn import random_projection

def run_random_project(X_scaled, y):
    transformer = random_projection.GaussianRandomProjection(n_components=X_scaled.shape[1] -1)
    X_rp = transformer.fit_transform(X_scaled)
    X_rp = pd.DataFrame(X_rp)
    X_rp.columns = [f'col{i+1}' for i in range(X_rp.shape[1])]

    data_rp_for_plot=pd.concat([X_rp, y], axis=1)

    for i in range(1, data_rp_for_plot.shape[1]):
        for j in range(i+1, data_rp_for_plot.shape[1]):
            for k in range(j+1, data_rp_for_plot.shape[1]):
                break
                plot_one(data_rp_for_plot, f'col{i}', f'col{j}', f'col{k}', 'Drug')
    return X_rp


X_scaled_em = run_em(X_scaled, y)

X_scaled_kmean = run_kmean(X_scaled, y)

X_scaled_kmean = run_kmean(X_scaled, y)


def plot_one(data, f1, f2, f3, target='Drug'):
    PLOT = go.Figure()
    for C in list(data[target].unique()):
        PLOT.add_trace(go.Scatter3d(x=data[data[target] == C][f1],
                                    y=data[data[target] == C][f2],
                                    z=data[data[target] == C][f3],
                                    mode='markers', marker_size=8, marker_line_width=1,
                                    name=f'{target} ' + str(C)))

    PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                       scene=dict(xaxis=dict(title=f1, titlefont_color='black'),
                                  yaxis=dict(title=f2, titlefont_color='black'),
                                  zaxis=dict(title=f3, titlefont_color='black')),
                       font=dict(family="Gilroy", color='black', size=12))
    PLOT.show()

plot_one(scaled_data, 'BP', 'Na_to_K', 'Cholesterol')
plot_one(scaled_data, 'Age', 'Na_to_K', 'Cholesterol')
plot_one(scaled_data, 'Age', 'Na_to_K', 'Sex')
plot_one(scaled_data, 'Age', 'Na_to_K', 'BP')

def run_pca(X, y):
    pca = decomposition.PCA()
    pca.fit(X)
    X_pca = pca.transform(X)

    X_pca=pd.DataFrame(X_pca)
    #X_pca = X_pca.copy()
    X_pca.index = X.index
    X_pca.columns = [f'col{i + 1}' for i in range(X_pca.shape[1])]
    data_pca_for_plot = pd.concat([X_pca, y], axis=1)

    # data_pca.columns = ['col1', 'col2', 'col3', 'col4', 'Drug']
    for i in range(1, data_pca_for_plot.shape[1]):
        for j in range(i + 1, data_pca_for_plot.shape[1]):
            for k in range(j + 1, data_pca_for_plot.shape[1]):
                plot_one(data_pca_for_plot, f'col{i}', f'col{j}', f'col{k}', 'Drug')
    return X_pca

x_pca = run_pca(X_scaled, y)

n_max = 100
n_start = 2
scores = []
inertia_list = np.empty(n_max)

for i in range(n_start,n_max):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x_pca)
    inertia_list[i] = kmeans.inertia_
    scores.append(silhouette_score(x_pca, kmeans.labels_))

n_max_shift = 2  # find maximum after this index of score
n_clusters = np.argmax(scores[n_max_shift:])+(n_start+n_max_shift) # it's my upgrade
plt.plot(range(0,n_max),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.axvline(x=n_clusters, color='blue', linestyle='--')
plt.ylabel('Inertia')
plt.show()

plt.plot(range(2,n_max), scores);
plt.title('Results KMeans')
plt.xlabel('n_clusters');
plt.axvline(x=n_clusters, color='blue', linestyle='--')
plt.ylabel('Silhouette Score');
plt.show()

x_rp = run_random_project(X_scaled, y)

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3,4]:

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)

plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Silhouetter Score')
plt.show()

score= silhouette_score(X,km.labels_,metric='euclidean')
print('Silhouetter Score: %.3f' % score)

x_sc_clusters =km.predict(X)
x_sc_clusters_centers =km.cluster_centers_

print(x_sc_clusters)

#ELBOW METHODU KISMI

from sklearn.cluster import  KMeans
kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)
sonuclar =[]

for i in range (1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('THE ELBOW METHOD')
plt.show()

x_sc_clusters =kmeans.predict(X)
x_sc_clusters_centers =kmeans.cluster_centers_
dist = [np.linalg.norm(x - y) for x, y in zip(X, x_sc_clusters_centers[x_sc_clusters])]

print('Elbow küme merkezine uzaklık durumu')
print(dist)

