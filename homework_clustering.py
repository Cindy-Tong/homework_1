import numpy as np

from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn import datasets
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = datasets.load_digits()
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

print(digits.images[1796])
print(digits.target[1796])
n_digits = len(np.unique(digits.target))

X=scale(digits.data)#标准化处理数据集，将每个属性的分布改为均值为0，标准差为1
x_compress=PCA(n_components=2).fit_transform(X)#数据集降为二维，用X来训练PCA模型，同时返回降维后的数据。
y_true = digits.target

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

y_pred = cluster.KMeans(n_clusters=10,random_state=9).fit_predict(X)
evaluate(y_true,y_pred,'k-means')
plt.subplot(331).set_title('k-means')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)#s为点的大小，c为点的颜色

y_pred = cluster.AffinityPropagation().fit_predict(X)
evaluate(y_true,y_pred,'AffinityPropafation')
plt.subplot(332).set_title('AffinityPropafation')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clf = MeanShift(bandwidth=1)
y_pred = clf.fit_predict(preprocessing.scale(X))
evaluate(y_true,y_pred,'MeanShift')
plt.subplot(333).set_title('MeanShift');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = cluster.SpectralClustering(n_clusters=n_digits, affinity='nearest_neighbors').fit_predict(X)
evaluate(y_true,y_pred,'SpectralClustering')
plt.subplot(334).set_title('SpectralClustering');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

clst=cluster.AgglomerativeClustering(n_clusters=n_digits,affinity="euclidean",linkage='ward')
y_pred = clst.fit_predict(X)
evaluate(y_true,y_pred,'AgglomerativeClustering')
plt.subplot(335).set_title('AgglomerativeClustering');
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = cluster.DBSCAN(eps = 4, min_samples = 4).fit_predict(X)
evaluate(y_true,y_pred,'DBSCAN')
plt.subplot(336).set_title('DBSCAN')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = mixture.GaussianMixture(n_components=10).fit_predict(X)
evaluate(y_true,y_pred,'GaussianMixture')
plt.subplot(337).set_title('GaussianMixture')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

print('-' * 50)
plt.show()