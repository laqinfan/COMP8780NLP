#
# Goal: Latent Semantic Analysis for text clustering
#
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import pandas
import numpy as np
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
from io import BytesIO
import base64
from plotly.tools import mpl_to_plotly

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def dataset_selection():
    op = ['20newsgroups','Random texts']
    options = [ {'label': v, 'value': v} for v in op ]
    return dcc.Dropdown(
        id = 'dropdown-data',
        options = options,
        )

def model_selection():
    op = ['kmeans-no-lsa','kmeans-lsa', 'lda']
    options = [ {'label': v, 'value': v} for v in op ]
    return dcc.Dropdown(
        id = 'dropdown-model',
        options = options,
        )

def image_selection():
    op = ['km-no-lsa.png','lsa.png']
    options = [ {'label': v, 'value': v} for v in op ]
    return dcc.Dropdown(
        id = 'dropdown-image',
        options = options,
        value=options[0]['value'],
        )

app.layout = html.Div(children=[
	html.H1(children='Latent Semantic Analysis for text clustering'),

    html.Div(children=[html.P(f'Select Dataset', style={'margin-left': '3px'}), dataset_selection(), ],
        # Step 5
        style={'width': '48%'},
    ),
    html.Div(children=[html.P(f'Select Clustering Model', style={'margin-left': '3px'}), model_selection()],
        # Step 5
        style={'width': '48%'},
    ),

    html.Div(className='row', children=[html.P(f'Number of Clusters', style={'margin-left': '3px'}),

	    html.Div(className='two columns',children = [ html.P(f'Start from:', style={'margin-left': '3px'}), dcc.Input(id='from', value='', type='number', placeholder='Start')
	    	]),
	    html.Div(className='two columns', children = [html.P(f'Stop at:', style={'margin-left': '3px'}), dcc.Input(id='to', value='', type='number', placeholder='Stop'),
	    	]),
		]),


    html.Div(children=[html.H3(f'Silhouette/Coherence Scores: ', style={'margin-left': '3px'}),],
        # Step 5
        style={'width': '48%'},
    ),
        html.Div(id='score_output'),

    html.Div(children = [ html.P(f'Enter Number of Clusters', style={'margin-left': '3px'}), dcc.Input(id='text-input1', value='', type='number')
    	], style={'width': '30%',}),

	html.Div(html.H3(f'Cluster Output:', style={'margin-left': '3px'}),id='model_output'),

	html.Button('Show cluster figure', id='button', style={'display': 'inline-block'}),
	html.Img(id='image-id'),

])
@app.callback(
	Output("image-id", "src"),
	[Input(component_id='dropdown-model', component_property='value'),
	 Input('button', 'n_clicks')])
def update_graph(model_value, n_clicks):
	if n_clicks:
		if model_value == 'kmeans-no-lsa':
			image = 'km-no-lsa.png'
			src=app.get_asset_url(image)
		elif model_value == 'kmeans-lsa':
			image = 'lsa.png'
			src=app.get_asset_url(image)
		else:
			src = None
		return src

@app.callback(
	Output(component_id='score_output', component_property='children'),
	[
		Input(component_id='from', component_property='value'),
		Input(component_id='to', component_property='value'),
		Input(component_id='dropdown-data', component_property='value'),
		Input(component_id='dropdown-model', component_property='value')
	]
)
def update_output_div(start, stop, data, model_value):

	pandas.set_option("display.max_colwidth", 200)
	doc = []
	categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'sci.electronics', 
    'sci.med',
    'talk.religion.misc',
	]

	if data == '20newsgroups':
		data = fetch_20newsgroups(categories=categories,shuffle = True, random_state = 1, remove = ('headers', 'footers', 'quoters'))
		doc = data.data
	else:
		with open( os.path.join("", "text.txt") ,"r") as fin:
		    for line in fin.readlines():
		        text = line.strip()
		        doc.append(text)

		# return html.H4('The predict house price is {}'.format(price))
		# return html.H3(children =['Cross validation score is {}'.format(score)], style={'color': 'green',})

	## Preprocessing the data
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

	term_list = []
	dict_out = {}

	if model_value== 'kmeans-no-lsa':
	    ###compute the silhouette score of different number of clusters
	    # range_clusters = [2,3,4,5,6,7,8,9,10]
	    silhouette_scores = []

	    for n in range(start, stop):
	        cluster = KMeans(n_clusters = n, random_state=10)
	        cluster_label = cluster.fit_predict(X_matrix)
	        silhouette = silhouette_score(X_matrix, cluster_label)
	        print("Cluser Number:", n, " silhouette_score:", round(silhouette,4))
	        silhouette_scores.append(round(silhouette, 4))
	        dict_out[n] = round(silhouette, 4)

	elif model_value== 'kmeans-lsa':
	    ###compute the silhouette score of different number of clusters
	    # range_clusters = [2,3,4,5,6,7,8,9,10]
	    dict_out = {}
	    silhouette_scores = []
	    for n in range(start, stop):
	        cluster = KMeans(n_clusters = n, random_state=10)
	        cluster_label = cluster.fit_predict(X)
	        silhouette = silhouette_score(X, cluster_label)
	        print("Cluser Number:", n, " silhouette_score:", round(silhouette,4))
	        silhouette_scores.append(round(silhouette, 4))
	        dict_out[n] = round(silhouette, 4)

	elif model_value == 'lda':
	    #### Topic modeling for clustering
	    dict_out = {}
	    dictionary = corpora.Dictionary(tokenized_doc)
	    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_doc]
	    # number_of_topics = [2,3,4,5,6,7,8,9,10]
	    for n in range(start, stop):
	        ldamodel = LdaModel(doc_term_matrix, num_topics=n, id2word = dictionary) 
	        m = CoherenceModel(model=ldamodel, texts=tokenized_doc, coherence='c_v')
	        covalue = m.get_coherence()
	        print("Cluster Number:", n, " Coherence Value:", round(covalue,4))
	        dict_out[n] = round(covalue, 4)

	# return html.H4('The predict house price is {}'.format(price))
	return html.H3(children =['{}'.format(dict_out)], style={'color': 'blue',})

@app.callback(
	Output(component_id='model_output', component_property='children'),
	[
		Input(component_id='text-input1', component_property='value'),
		Input(component_id='dropdown-data', component_property='value'),
		Input(component_id='dropdown-model', component_property='value')
	]
)
def update_output_div(cluster,data, model_value):

	pandas.set_option("display.max_colwidth", 200)
	doc = []
	categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'sci.electronics', 
    'sci.med',
    'talk.religion.misc',
	]

	if data == '20newsgroups':
		data = fetch_20newsgroups(categories=categories,shuffle = True, random_state = 1, remove = ('headers', 'footers', 'quoters'))
		doc = data.data
	else:
		with open( os.path.join("", "text.txt") ,"r") as fin:
		    for line in fin.readlines():
		        text = line.strip()
		        doc.append(text)

		# return html.H4('The predict house price is {}'.format(price))
		# return html.H3(children =['Cross validation score is {}'.format(score)], style={'color': 'green',})

	## Preprocessing the data
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
	dict_term = {}

	if model_value== 'kmeans-no-lsa':

	    ###select the clsuter number of highest silhousette score
	    ###KMeans to cluser the articles
	    km = KMeans(n_clusters = cluster, random_state=10)
	    km.fit(X_matrix)

	    term_list = []

	    centers = km.cluster_centers_.argsort()[:, ::-1]
	    clusters = km.labels_.tolist()

	    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(X_matrix)
	    plt.figure(figsize=(3,3))
	    plt.scatter(embedding[:, 0], embedding[:, 1], 
	    c = clusters,
	    s = 5, # size
	    edgecolor='none'
	    )
	    # plt.show()
	    plt.savefig('assets/km-no-lsa.png', dpi=200) #save figure as hr.png
	    image = Image.open('assets/km-no-lsa.png')
	    image.show()

	    ###get each word, which is actually a topic
	    topics = vector.get_feature_names()
	    for i in range(cluster):
	        print("Cluster %d:" % i, end='')
	        for ind in centers[i, :8]:
	            print(' %s' % topics[ind], end='')
	            term_list.append(topics[ind])
	        print()
	       	dict_term[i] = term_list
	        term_list = []

	elif model_value== 'kmeans-lsa':
	    ###select the clsuter number of highest silhousette score
	    ###KMeans to cluser the articles
	  
	    km = KMeans(n_clusters = cluster, random_state=10)
	    km.fit(X)
	    dict_term = {}
	    term_list = []

	    centroids = svd_model.inverse_transform(km.cluster_centers_)
	    centers = centroids.argsort()[:, ::-1]
	    clusters = km.labels_.tolist()
	    topics = U*Sigma
	    embedding = umap.UMAP(n_neighbors=100, min_dist=0.5, random_state=12).fit_transform(topics)
	    plt.figure(figsize=(3,3))
	    plt.scatter(embedding[:, 0], embedding[:, 1], 
	    c = clusters,
	    s = 5, # size
	    edgecolor='none'
	    )
	    # plt.show()
	    plt.savefig('assets/lsa.png', dpi=200) #save figure as hr.png
	    image = Image.open('assets/lsa.png')
	    image.show()
	    ###get each word, which is actually a topic
	    topics = vector.get_feature_names()
	    for i in range(cluster):
	        print("Cluster %d:" % i, end='')
	        for ind in centers[i, :8]:
	            print(' %s' % topics[ind], end='')
	            term_list.append(topics[ind])
	        print()
	        dict_term[i] = term_list
	        term_list = []


	elif model_value == 'lda':
	    # generate LDA model
	    dict_term = {}
	    term_list = []
	    vector_count = CountVectorizer()
	    X_transform = vector_count.fit_transform(news_df['clean_doc'])
	    terms = vector_count.get_feature_names()
	    lda = LatentDirichletAllocation(
	        n_components=cluster, max_iter=5, 
	        learning_method='online', random_state=0)
	    lda.fit(X_transform)

	    for index, topic in enumerate(lda.components_):
	        terms_topic = zip(terms, topic)
	        sorted_terms = sorted(terms_topic, key= lambda x:x[1], reverse=True)[:8]
	        print("Cluster "+str(index)+": ", end='')
	        for t in sorted_terms:
	            print(t[0], " ", end='')
	            term_list.append(t[0])
	        print()
	        dict_term[index] = term_list
	        term_list = []

	# return html.H4('The predict house price is {}'.format(price))
	return html.H3(children =['{}'.format(dict_term)], style={'color': 'green',})

if __name__ == '__main__':
	app.run_server(debug=True)