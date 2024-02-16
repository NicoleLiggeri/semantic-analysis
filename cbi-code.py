import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import spacy
from spacy.lang.it.stop_words import STOP_WORDS
from nltk.stem import SnowballStemmer
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt


# Load Italian language model
nlp = spacy.load('it_core_news_sm')

# Get the Italian stopwords
italian_stopwords = set(STOP_WORDS)

def preprocess_italian_text(text):
    # Process the text using the Italian language model
    nlp = spacy.load("it_core_news_sm")

    # Process the text using the Italian language model
    doc = nlp(text)

    # Extract lemmatized tokens from the processed text, remove stopwords, convert to lowercase, and filter out punctuation
    preprocessed_tokens = [token.lemma_.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha and token.lemma_.lower() != "il" and token.lemma_.lower() != "ci"]
    preprocessed_tokens= ' '.join(preprocessed_tokens)
    doc = nlp(preprocessed_tokens)
    new = [token.lemma_.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha and token.lemma_.lower() != "il" and token.lemma_.lower() != "ci"]
    return ' '.join(new)



# Define a custom function for LogTF
def log_tf(term_freq_matrix):
    epsilon = 1e-12  # Small epsilon value to avoid taking log of zero
    return 1 + np.log(term_freq_matrix + epsilon)

# Load data from CSV file
data = pd.read_csv("C:/Users/Nicole/Desktop/dialoghiLeucò.csv", sep=";")

# Preprocess the text in the "testo" column
data['preprocessed_testo'] = data['"testo"'].apply(lambda x: preprocess_italian_text(x))



# Define the number of clusters
num_clusters = 4

# Create a pipeline for LSA and clustering
pipeline = make_pipeline(
    TfidfVectorizer(),
    TruncatedSVD(n_components=num_clusters),
    Normalizer(copy=False),
    KMeans(n_clusters=num_clusters)
)

# Fit the pipeline to the preprocessed text
pipeline.fit(data['preprocessed_testo'])
data['cluster'] = pipeline.predict(data['preprocessed_testo'])
data.to_csv('C:/Users/Nicole/Desktop/dialoghiProcessati.csv', index=False)


G = nx.Graph()

# Add Clusters as Nodes
for cluster_id in range(num_clusters):
    G.add_node(f"Cluster {cluster_id}")

# Add Documents as Nodes and Connect to Clusters
for doc_id, (cluster_id, title) in enumerate(zip(data['cluster'], data['"titolo"'])):
    doc_node_name = f"Document {doc_id}"
    G.add_node(doc_node_name,title=title)  # Add title as node attribute
    G.add_edge(doc_node_name, f"Cluster {cluster_id}")

node_colors = []
for node in G.nodes():
    if node.startswith("Cluster"):
        node_colors.append('red')  # Assign green color to document nodes
    else:
        node_colors.append('green')

nx.draw(G, with_labels=True, node_color=node_colors)
plt.show()

