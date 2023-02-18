#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple command-line example for Custom Search.
Command-line application that does a search.
"""

# __author__ = Diane Gu (xg2399), Hanna Gao (qg2205)"

import pprint

from googleapiclient.discovery import build

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

import sys

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import defaultdict

from itertools import permutations


# Ask the user to mark the results as relevant or non-relevant
# Parameter:
#   res: dictionary of top10 query results
# Returns:
#  Two list of relevant and non-relevant documents in a set
def userInput(res, API_key, engine_id, query, precision_boundary):
    
    # Initialize 
    rele_docs = []
    nonrele_docs = []
    
    # Retrieve top 10 results
    top10 = res['items'][:10]
    print("Parameters: \n \
    Client key  = ", API_key,
    "\nEngine key  = ", engine_id,
    "\nQuery       = ", query,
    "\nPrecision   = ", precision_boundary,
   "\nGoogle Search Results:")
    print('======================')

    
    # Show the url, title, and description  of top 10 results to the user
    for i in range(len(top10)):
        url = top10[i]['formattedUrl']
        title = top10[i]['title']
        snippet = top10[i]['snippet']


        print("Result ", i+1)
        print('[')
        print(' URL: ',url)
        print(' Title: ', title)
        print(' Summary', snippet)
        print(']')
        print()
        
        # User input 
        rele = input('Relevant (Y/N)?')

        # Store and classify the results' title and snippet based on user input
        if rele=='Y':
            print(title+snippet)
            rele_docs.append(title+snippet)
        else:
            nonrele_docs.append(title+snippet)
    
    return (rele_docs,nonrele_docs)


# Return a list of stop words by reading from a text file
def stop_word(filename):
    with open(filename) as f:
        contents = f.read()
    contents = contents.split()
    return contents
        

# Transform all documents into tf-idf vectors using sklearn TfidfVectorizer package
# Return the tf-idf vectors of the collection and the corresponding words
def tfidfTransform(corpus, stop_words = None):
    vectorizer = TfidfVectorizer(stop_words = stop_words)
    X = vectorizer.fit_transform(corpus)
    Y = X.toarray()
    return Y, vectorizer.get_feature_names_out()



# Return the top 10 results of the query on Google
def web_search(query, API_key, engine_id):
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.

    service = build(
        "customsearch", "v1", developerKey=API_key
    )
    
    res = (
        service.cse()
        .list(
            q=query,
            cx=engine_id,
        )
        .execute()
    )
    return res


# Rochhio algroithm
# Parameters: 
#   collection_vector: a list of tf-idf vectors of the collection
#   rel_count: number of relevant documents
#   nrel_count: number of non-relevant documents
# Returns:
#   new_query_vector: tf-idf vector of the new query
def rocchio(collection_vector, rel_count, nrel_count):
    
    # Initialize constants
    alpha = 1.0
    beta = 0.75
    gamma = 0.15
    
    # Extract the tf-idf vectors for query
    query_vector = collection_vector[0, :]
    
    # Calculate the sum the tf-idf vectors for relevant and non-relevant documents respectively
    relevant_vector = np.sum(collection_vector[1:1+rel_count, :], axis = 0)
    non_relevant_vector = np.sum(collection_vector[1+rel_count:, :], axis = 0)
    
    # Calculate the tf-idf vector of new query using Rochhio algorithm
    new_query_vector = alpha*query_vector + beta*relevant_vector/rel_count - gamma*non_relevant_vector/nrel_count
    
    # Set all negative term in the new query vector to 0
    new_query_vector[new_query_vector<0] = 0
    
    return new_query_vector
  


# Find the words to be augumented into the query
# Parameters:
#   new_query_vector: tf-idf vector of the new query
#   wordDict: corresponding word to each element in the tf-idf vector
#   query: string of original query
# Return: 
#   new_words: words to be augumented into the query
def augmentWord(new_query_vector, wordDict, query, bigramDict):
    
    # Combine the tf-idf vector with the corresponding word
    vec_word = zip(new_query_vector, wordDict)

    # Sort vec_word from high to low
    sort_vec_word = sorted(vec_word, reverse=True, key=lambda x: x[0])
    query = query.lower().split(" ")
    
    # Find the top 2 new words to be added into the query 
    count = 0
    i = 0
    new_words = []
    while count < 2:
        if sort_vec_word[i][1] in query:
            i+=1
        else:
            new_words.append(sort_vec_word[i][1])
            i+=1
            count+=1
    
    new_query = compute_ordering(query + new_words, bigramDict)
    
    return new_query




def compute_ordering(query, bigramDict):
    perms = permutations(query)
    
    best_ordering = ''
    max_freq = float('-inf')
    
    for i in perms:
        i_bigram = list(nltk.bigrams(i))
        freq = 0

        for bigram in i_bigram:
            freq += bigramDict[bigram]

        if freq > max_freq:
            max_freq = freq
            best_ordering = i
    return best_ordering
            
     

# Compute the bigram from the corpus and update the bigram count dictionary
def compute_bigram(bigramDict, corpus):
    for sentence in corpus:
        words = word_tokenize(sentence.lower())
        for bigram in list(nltk.bigrams(words)):
            bigramDict[bigram] += 1
    return bigramDict
        
    

# We use Rocchio Algroithm to implement Query Expansion
def main():
    
    # Initialize command line parameters
    API_key = sys.argv[1]
    engine_id = sys.argv[2]
    precision_boundary = float(sys.argv[3])
    query = sys.argv[4]
    
    # Initialize precision
    precision = 0.0    
    
    # Define stop words
    stop_words = stop_word('proj1-stop.txt')
    
    # Initialize bigram count dictionary
    bigramDict = defaultdict(int)
    
    # Query expansion
    while precision < precision_boundary: 
        
        # Retrieve the top 10 results to the current query
        res = web_search(query, API_key, engine_id)
        
        # User define relevenace of the top 10 results
        relevant, non_relevant = userInput(res, API_key, engine_id, query, precision_boundary)
        
        # Calculate precision of this iteration
        relevant_count = len(relevant)
        non_relevant_count = len(non_relevant)
        new_precision = relevant_count/(relevant_count + non_relevant_count)
        
        # Break the query expansion process if the precision is above or equal 
        # to the Threshold or if the precision is zero
        if new_precision == 0 or precision >= precision_boundary:
            break

        # Transform query, relevant, and non-relevant results into tf-idf vectors
        collection = [query] + relevant + non_relevant
        collection_vector, wordDict = tfidfTransform(collection, stop_words)
        
        # Calculate the bigram from collection
        bigramDict = compute_bigram(bigramDict, collection)
        
        # Expand the query using Rocchio algorithm
        new_query_vector = rocchio(collection_vector, relevant_count, non_relevant_count)
        # query = query + " " + " ".join(augmentWord(new_query_vector,wordDict, query))
        new_query = augmentWord(new_query_vector, wordDict, query, bigramDict)

        query = " ".join(new_query)
        print('The New Query is: ', query)
        
        



if __name__ == "__main__":
    main()

    
