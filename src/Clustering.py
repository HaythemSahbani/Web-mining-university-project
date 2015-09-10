
from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import json



class Clustering:

    def __init__(self, sample_cluster=[], n_features=1000, ngram=1):
        self.n_features = n_features
        self.ngram = ngram
        self.sample_cluster = sample_cluster

        tfidf_vect= TfidfVectorizer(max_df=0.95, max_features=self.ngram*self.n_features,
                                     min_df=2, stop_words='english', ngram_range=(1, self.ngram))

        self.tfidf_vect = tfidf_vect
        pass

    def present_clusters(self, best_k, X):
        """
        Draw clusters, centroids, bordrers and points that belongs to each cluster
        :param best_k: number of cluster
        :param X: Matrix of weighted tweets
        :return:
        """

        reduced_X = PCA(n_components=2).fit_transform(X.toarray())
        kmeans = KMeans(init='k-means++', n_clusters=best_k, n_init=1)
        kmeans.fit(reduced_X)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        step = .02     # point in the mesh [min_x, max_y]x[min_y, max_y].
        # Plot the decision boundary. For that, we will assign a color to each
        min_x, max_x = reduced_X[:, 0].min() + 0.1, reduced_X[:, 0].max() - 0.1
        min_y, max_y = reduced_X[:, 1].min() + 0.1, reduced_X[:, 1].max() - 0.1


        xx, yy = np.meshgrid(np.arange(min_x, max_x, step), np.arange(min_y, max_y, step))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_X[:, 0], reduced_X[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on tweets (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def best_kmeans(self, best_k, t_list):
        """
        Clustering the tweets into best_k topics
        :param best_k: number of clusters
        :param t_list: list of tweets
        :return:
        """

        print("#####################   Clustering with best matching k=%d  ####################" % best_k)
        X = self.tfidf_vect.fit_transform(t_list)

        # Do the actual clustering
        km = KMeans(n_clusters=best_k, init='k-means++', max_iter=100, n_init=1)

        model = km.fit(X)

        self.sample_cluster = model.predict(X)


        print("Top terms per cluster:")
        centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = self.tfidf_vect.get_feature_names()

        for i in range(best_k):
            print(" Cluster %s:" % str(i+1), end='')
            if self.ngram == 1:
                for ind in centroids[i, :5]:
                    print(' %s,' % terms[ind], end='')
                print(" ")
            else:
                cluster_sample = []
                for ind in centroids[i]:
                    for term in terms[ind].split():
                        cluster_sample.append(term)
                print(list(set(cluster_sample[:20])))

            print()

        self.present_clusters(best_k, X)

    def k_gap(self, ks, logWks, logWkbs, sk):

        Gap = list(np.array(logWkbs)-np.array(logWks))

        #draw gap(k)
        plt.plot(ks, Gap, 'o-')
        plt.show()


        for i in range(len(Gap)-1):
            Gap[i] = Gap[i]-(Gap[i+1]-sk[i+1])
            if (Gap[i] >= 0):
                break

        print("best k found by gap statistic =%d" % ks[i])
        return ks[i]

    def generate_ref(self, nref, X):
        shape = X.shape
        maxs = np.amax(X, axis=0)
        mins = np.amin(X, axis=0)

        dstr = np.matrix(np.diag(maxs-mins))
        ref = np.random.random_sample(size=(shape[0], shape[1], nref))
        for i in range(nref):
            ref[:, :, i] = ref[:, :, i]*dstr+mins
        return ref

    def gap_statistic(self, t_list, kmin=2, kmax=10):
        """
        Detect the optimal number of clusters by varying the number between kmin and kmax and select the best result
        :param t_list: list of tweets
        :param kmin: minimum number of clusters
        :param kmax: maximum number of clusters
        :return:
        """

        self.tfidf_vect.max_features = self.n_features
        X = self.tfidf_vect.fit_transform(t_list)

        #Create nref reference Dataset
        nref = 10
        X = X.toarray()
        data_ref = self.generate_ref(nref, X)

        # Dispersion for real distribution
        #Varying k between kmin and kmax...
        ks = range(kmin, kmax)
        Wks = np.zeros(len(ks))
        Wkbs = np.zeros(len(ks))
        sk = np.zeros(len(ks))

        for indk, k in enumerate(ks):
            km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
            modelX = km.fit(X)

            Wks[indk] = np.log(modelX.inertia_)

            # Compute the standard deviation with nref reference dataset

            BWkbs = np.zeros(nref)
            for i in range(nref):
                Xref = data_ref[:, :, i]
                modelXref = km.fit(Xref)
                BWkbs[i] = np.log(modelXref.inertia_)
            Wkbs[indk] = sum(BWkbs)/nref
            sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/nref)

        sk = sk*np.sqrt(1+1/nref)
        #get the best k
        best_k = self.k_gap(ks, Wks, Wkbs, sk)

        return best_k

    def get_followers(self, k, t_list):
        """
        Get list of followers of each topic
        :param k: number of clusters
        :param t_list: list of tweets
        :return:
        """
        followers = {}
        for ind, k in enumerate(self.sample_cluster):
            l = [token for token in t_list[ind].split() if token.startswith("@")]
            l1 = followers.setdefault(k, list()) + l
            followers[k] = l1
        return followers

    def set_tweet_topic(self, json_tweets):
        """
        Update tweets with new assignment of topics
        :param file:contain the json_tweet
        :return:
        """
        for ind, i in enumerate(self.sample_cluster):
            json_tweets[ind]["topic"] = i
