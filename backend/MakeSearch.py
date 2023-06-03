import re

from collections import defaultdict
import math

import ujson

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from fastapi_pagination import Page, paginate

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

with open('../crawler/res.json', 'r') as doc: scraper_results=doc.read()

publicationName = []
pubURL = []
pubCUAuthor = []
pubDate = []
dict_search = ujson.loads(scraper_results)
array_length = len(dict_search)

for search_item in dict_search:
    publicationName.append(search_item["name"])
    pubURL.append(search_item["pub_url"])
    pubCUAuthor.append(search_item["cu_author"])
    pubDate.append(search_item["date"])


stemmer = PorterStemmer()
publication_list_after_stem = []
pub_list = []

stop_words = set(stopwords.words("english"))


def stopWordClean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
    processed_text = " ".join(words)
    
    return processed_text



for file in publicationName:
    preprocess_text = stopWordClean(file)
    publication_list_after_stem.append(preprocess_text)
    pub_list.append(file)



def makeInvertedIndex(documents):
    list_inverted_index = defaultdict(list)
    
    for doc_id, doc in enumerate(documents):
        terms = doc.split()
        for term in terms:
            list_inverted_index[term].append(doc_id)
    
    return list_inverted_index


list_inverted_index = makeInvertedIndex(publication_list_after_stem)

def search(query=''):
    totalData = len(dict_search)
    data = dict_search
    if query == '':
        return {
            'data': data,
            'totalData': totalData
        }
    processed_query = stopWordClean(query)
    query_terms = processed_query.split()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(publication_list_after_stem)
    relevant_docs = set()
    for term in query_terms:
        if term in list_inverted_index:
            relevant_docs.update(list_inverted_index[term])

    relevant_docs = list(relevant_docs)
    tfidf_query = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(tfidf_query, tfidf_matrix[relevant_docs])
    sorted_docs = sorted(zip(relevant_docs, cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    data = [dict_search[idx] for idx, _ in sorted_docs]
    return {
            'data': data,
            'totalData': len(data)
        }

