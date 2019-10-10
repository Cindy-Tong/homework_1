# homework_1实验报告
## 实验要求
针对以下两个数据集应用指定的聚类方法，并测试聚类效果  
### 数据集  
1 sklearn.datasets.load_di gits,visit [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)  
![Image text](https://github.com/Cindy-Tong/homework_1/blob/master/image-folder/digits_attributes.PNG)  
2 sklearn.datasets.fetch_20newsgroups,visit [20newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups)  
![Image text](https://github.com/Cindy-Tong/homework_1/blob/master/image-folder/20newsgroups.PNG)  
### 聚类方法
K-Means  
Affinity propagation  
Mean-shift  
Spectral clustering  
Ward hierarchical clustering  
Agglomerative clustering  
DBSCAN  
Gaussian mixtures  
details visit [clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering)  
## 实验过程  
导入numpy并从sklearn中导入需要用到的包
```
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
```
导入digits数据集并获得数据类别的个数，对数据集应用PCA进行处理降维到二维平面便于聚类可视化
```
digits = datasets.load_digits()
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
n_digits = len(np.unique(digits.target))

X=scale(digits.data)#标准化处理数据集，将每个属性的分布改为均值为0，标准差为1
x_compress=PCA(n_components=2).fit_transform(X)#数据集降为二维，用X来训练PCA模型，同时返回降维后的数据。
y_true = digits.target
```
定义评估方法，方法返回Normalized Mutual Information(NMI)、Homogeneity、Completeness三组数据对样本真实值和预测值之间的评估结果  
```
def evaluate(labels_true,labels_pred,name):
    print('%-20s\t%.3f\t%.3f\t%.3f'%(
        name,
        metrics.v_measure_score(labels_true,labels_pred),
        metrics.homogeneity_score(labels_true,labels_pred),
        metrics.completeness_score(labels_true,labels_pred)
    ))
```
调用各聚类方法对数据集X进行预测并评估
```
y_pred = cluster.KMeans(n_clusters=10,random_state=9).fit_predict(X)
evaluate(y_true,y_pred,'k-means')
plt.subplot(331).set_title('k-means')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred*100)#s为点的大小，c为点的颜色

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
evaluate(y_true,y_pred,'WardHierarchical')
plt.subplot(335).set_title('WardHierarchical')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

i = 0
for linkage in ('average','complete'):
    clst = cluster.AgglomerativeClustering(linkage=linkage,n_clusters=n_digits,affinity='canberra')
    y_pred = clst.fit_predict(X)
    evaluate(y_true,y_pred,linkage+'Agglomerative')
    plt.subplot(336+i).set_title(linkage+'Agglomerative')
    i = i+1
    plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.DBSCAN(eps = 4.3, min_samples = 6).fit_predict(X)
evaluate(y_true,y_pred,'DBSCAN')
plt.subplot(338).set_title('DBSCAN')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

y_pred = mixture.GaussianMixture(n_components=10).fit_predict(X)
evaluate(y_true,y_pred,'GaussianMixture')
plt.subplot(339).set_title('GaussianMixture')
plt.scatter(x_compress[:,0],x_compress[:,1],c=y_pred)

print('-' * 50)
plt.show()
```

评估结果如下所示：  
