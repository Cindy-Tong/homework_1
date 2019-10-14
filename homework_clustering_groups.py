import numpy as np

from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn import datasets
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'rec.motorcycles',
    'comp.graphics',
    'sci.space',
]

newsdata = datasets.fetch_20newsgroups(subset='all', categories=categories)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsdata.data)
lsa = TruncatedSVD(2)
X = lsa.fit_transform(X)
x_compress = PCA(n_components=2).fit_transform(X)
y_true = newsdata.target
true_k = np.unique(y_true).shape[0]

def evaluate(labels_true,labels_pred,name):
    print('%-20s\t%.3f\t%.3f\t%.3f'%(
        name,
        metrics.v_measure_score(labels_true,labels_pred),
        metrics.homogeneity_score(labels_true,labels_pred),
        metrics.completeness_score(labels_true,labels_pred)
    ))

print('-'*50)
print('%-20s\t%s\t%s\t%s'%('','NMI','HOMO','COMP'))
plt.figure(figsize=(10, 10))#自定义画布大小

y_pred = cluster.KMeans(n_clusters=true_k,random_state=9).fit_predict(X)
evaluate(y_true,y_pred,'k-means')
plt.subplot(331).set_title('k-means')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)#s为点的大小，c为点的颜色

y_pred = cluster.AffinityPropagation(damping=0.88, preference=-3000).fit_predict(X)
evaluate(y_true,y_pred,'AffinityPropafation')
plt.subplot(332).set_title('AffinityPropafation')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clf = MeanShift(bandwidth=0.0001, bin_seeding=True)
y_pred = clf.fit_predict(preprocessing.scale(X))
evaluate(y_true,y_pred,'MeanShift')
plt.subplot(333).set_title('MeanShift');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = cluster.SpectralClustering(n_clusters=true_k, affinity='nearest_neighbors').fit_predict(X)
evaluate(y_true,y_pred,'SpectralClustering')
plt.subplot(334).set_title('SpectralClustering');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clst=cluster.AgglomerativeClustering(n_clusters=true_k,affinity="euclidean",linkage='ward')
y_pred = clst.fit_predict(X)
evaluate(y_true,y_pred,'WardAgglomerative')
plt.subplot(335).set_title('WardAgglomerative');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clst=cluster.AgglomerativeClustering(n_clusters=true_k,affinity="canberra",linkage='average')
y_pred = clst.fit_predict(X)
evaluate(y_true,y_pred,'AverageAgglomerative')
plt.subplot(336).set_title('AverageAgglomerative');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clst=cluster.AgglomerativeClustering(n_clusters=true_k,affinity="canberra",linkage='complete')
y_pred = clst.fit_predict(X)
evaluate(y_true,y_pred,'CompleteAgglomerative')
plt.subplot(337).set_title('CompleteAgglomerative');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = cluster.DBSCAN(eps = 0.005, min_samples = 2).fit_predict(X)
evaluate(y_true,y_pred,'DBSCAN')
plt.subplot(338).set_title('DBSCAN')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = mixture.GaussianMixture(n_components=true_k).fit_predict(X)
evaluate(y_true,y_pred,'GaussianMixture')
plt.subplot(339).set_title('GaussianMixture')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

print('-' * 50)
plt.show()