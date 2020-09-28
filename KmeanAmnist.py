import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)
fig,axs = plt.subplots(3,3, figsize=(12,12))
plt.gray()

for i,ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title("Number {}".format(y_train[i]))



plt.show()
X = x_train.reshape(len(x_train),-1)
Y = y_train.reshape(len(y_train),-1)
X = X.astype(float) / 255.



from sklearn.cluster import MiniBatchKMeans

n_digits = len(np.unique(y_test))
print(n_digits)

kmean = MiniBatchKMeans(n_clusters=n_digits)
kmean.fit(X)


def infer_labels(kmeans,actual_labels):
    infered_labels={}
    for i in range(kmeans.n_clusters):

        labels = []
        index = np.where(kmeans.labels_ == i)
        labels.append(actual_labels[index])
        if len(labels[0])==1:
            count = np.bincount(labels[0])
        else:
            count = np.bincount(np.squeeze(labels))
        if np.argmax(count) in infered_labels:
            infered_labels[np.argmax(count)].append(i)
        else:
            infered_labels[np.argmax(count)] = [i]
        #print(labels)
        #print("Cluster={}  Labels = {}".format(i,np.argmax(count)))
    return infered_labels

def infer_data_label(X_labels,cluster_labels):

    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    for i,cluster in enumerate(X_labels):
        for key,value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
    return predicted_labels
    #count = np.bincount(np.squeeze(labels))


cluster_labels = infer_labels(kmean,Y)
X_cluster = kmean.predict(X)
predicted_labels = infer_data_label(X_cluster,cluster_labels)

print(predicted_labels[:20])
print(Y[:20].reshape(1,-1))
# array = np.zeros((1,3))
# print(array.shape)
# print(np.bincount(np.squeeze(array).astype(np.uint8)))

from sklearn import metrics

def calaculate_metrics(estimator,data,labels):
    print("Number of Clusters : ",estimator.n_clusters)
    print("Inertia : ",estimator.inertia_)
    print("Homogenity : ",metrics.homogeneity_score(labels.flatten(),estimator.labels_))

clusters = [10,16,24,36,144,256]

# for ctr in clusters:
#     estimator = MiniBatchKMeans(n_clusters=ctr)
#     estimator.fit(X)

#     calaculate_metrics(estimator,X,Y)

#     cluster_label = infer_labels(estimator,Y)
#     predicted_Y = infer_data_label(estimator.labels_,cluster_label)

#     print("Accuracy Score : {}\n".format(metrics.accuracy_score(Y,predicted_Y)))

X_test = x_test.reshape(len(x_test),-1)
X_test = X_test.astype(float) / 255.0

kmean = MiniBatchKMeans(n_clusters=256)
kmean.fit(X)
cluster_label = infer_labels(kmean,Y)

test_cluster = kmean.predict(X_test)
predicted_labels = infer_data_label(test_cluster,cluster_label)

print("Testing Accuracy : ",metrics.accuracy_score(y_test,predicted_labels))
centroids = kmean.cluster_centers_

images = centroids.reshape(256,28,28)
images*=255
images = images.astype(np.uint8)

cluster_labels = infer_labels(kmean,Y)

fig,axs = plt.subplots(3,3, figsize=(20,20))
plt.gray()
ax.axis('off')

for i ,ax in enumerate(axs.flat):
    for key,value in cluster_labels.items():
        if i in value:
            ax.set_title("Infered Label: {}".format(key))
    ax.matshow(images[i])
    ax.axis('off')
plt.show()