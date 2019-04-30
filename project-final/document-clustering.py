import numpy as np 
import pandas, os, sys, umap
from PIL import Image
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'sci.electronics', 
    'sci.med',
    'talk.religion.misc',
]

#fetch 20newsgroups data
data = fetch_20newsgroups(categories=categories,shuffle = True, random_state = 1, remove = ('headers', 'footers', 'quoters'))
doc = data.data
names = data.target_names
print(names)


######################## Preprocessing the data ########################
##Remove the punctuations, numbers, and special characters

news_df = pandas.DataFrame({'document':doc})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
#remove short words which are not meaningful in the contexts
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# make all text lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

##Tokenize the texts
##Remove stop words
stop_words = stopwords.words('english')

# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc

### Document Term Matrix---tfidf
vector = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True)
X_matrix = vector.fit_transform(news_df['clean_doc'])
print(X_matrix.shape)

### reduce dimentionality using SVD
n_component = 100
svd_model = TruncatedSVD(n_components=n_component)
U, Sigma, VT = randomized_svd(X_matrix, n_components=n_component, n_iter=100, random_state=122)

normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd_model, normalizer)
X = lsa.fit_transform(X_matrix)

if sys.argv[1] == 'kmeans-no-lsa':
    ###compute the silhouette score of different number of clusters
    range_clusters = [2,3,4,5,6,7,8,9,10]
    silhouette_scores = []
    for n in range_clusters:
        cluster = KMeans(n_clusters = n, random_state=10)
        cluster_label = cluster.fit_predict(X_matrix)
        silhouette = silhouette_score(X_matrix, cluster_label)
        print("Cluser Number:", n, " silhouette_score:", round(silhouette,4))
        silhouette_scores.append(round(silhouette, 4))

    ###select the clsuter number of highest silhousette score
    ###KMeans to cluser the articles
    var = input("Please input the number of clusters: ")
    km = KMeans(n_clusters = int(var), random_state=10)
    km.fit(X_matrix)

    centers = km.cluster_centers_.argsort()[:, ::-1]
    ###get each word, which is actually a topic
    topics = vector.get_feature_names()
    for i in range(int(var)):
        print("Cluster %d:" % i, end='')
        for ind in centers[i, :8]:
            print(' %s' % topics[ind], end='')
        print()

    centers = km.cluster_centers_.argsort()[:, ::-1]
    clusters = km.labels_.tolist()

    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_matrix)
    plt.figure(figsize=(7,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = clusters,
    s = 10, # size
    edgecolor='none'
    )
    # plt.show()
    plt.savefig('km-no-lsa.png', dpi=200) #save figure as hr.png
    image = Image.open('km-no-lsa.png')
    image.show()
elif sys.argv[1] == 'kmeans-lsa':
    ###compute the silhouette score of different number of clusters
    range_clusters = [2,3,4,5,6,7,8,9,10]
    silhouette_scores = []
    for n in range_clusters:
        cluster = KMeans(n_clusters = n, random_state=10)
        cluster_label = cluster.fit_predict(X)
        silhouette = silhouette_score(X, cluster_label)
        print("Cluser Number:", n, " silhouette_score:", round(silhouette,4))
        silhouette_scores.append(round(silhouette, 4))

    ###select the clsuter number of highest silhousette score
    ###KMeans to cluser the articles
    var = input("Please input the number of clusters: ")
    km = KMeans(n_clusters = int(var), random_state=10)
    km.fit(X)

    centroids = svd_model.inverse_transform(km.cluster_centers_)
    centers = centroids.argsort()[:, ::-1]
    ###get each word, which is actually a topic
    topics = vector.get_feature_names()
    for i in range(int(var)):
        print("Cluster %d:" % i, end='')
        for ind in centers[i, :8]:
            print(' %s' % topics[ind], end='')
        print()
    clusters = km.labels_.tolist()
    topics = U*Sigma
    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(topics)
    plt.figure(figsize=(7,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = clusters,
    s = 10, # size
    edgecolor='none'
    )
    # plt.show()
    plt.savefig('lsa.png', dpi=200) #save figure as hr.png
    image = Image.open('lsa.png')
    image.show()

elif sys.argv[1] == 'Hierarchical' or sys.argv[1] == 'hr':
    #### Hierarchical clustering
    distance = 1 - cosine_similarity(X_matrix)
    linkage_matrix = ward(distance)
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    topics = vector.get_feature_names()
    ax = dendrogram(linkage_matrix);
    plt.tight_layout()
    plt.savefig('hr.png', dpi=200) #save figure as hr.png
    image = Image.open('hr.png')
    image.show()

elif sys.argv[1] == 'lda':
    #### Topic modeling for clustering
    dictionary = corpora.Dictionary(tokenized_doc)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_doc]
    number_of_topics = [2,3,4,5,6,7,8,9,10]
    for n in number_of_topics:
        ldamodel = LdaModel(doc_term_matrix, num_topics=n, id2word = dictionary) 
        m = CoherenceModel(model=ldamodel, texts=tokenized_doc, coherence='c_v')
        covalue = m.get_coherence()
        print("Cluster Number:", n, " Coherence Value:", round(covalue,4))

    # generate LDA model
    var = input("Please input the number of clusters: ")
    vector_count = CountVectorizer()
    X_transform = vector_count.fit_transform(news_df['clean_doc'])
    terms = vector_count.get_feature_names()
    lda = LatentDirichletAllocation(
        n_components=int(var), max_iter=5, 
        learning_method='online', random_state=0)
    lda.fit(X_transform)

    for index, topic in enumerate(lda.components_):
        terms_topic = zip(terms, topic)
        sorted_terms = sorted(terms_topic, key= lambda x:x[1], reverse=True)[:8]
        print("Cluster "+str(index)+": ", end='')
        for t in sorted_terms:
            print(t[0], " ", end='')
        print()





