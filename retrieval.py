# Name: Sai Aakarsh reddy Koppula
# netId: skoppu3
# UIN: 660014307


import os
import pandas as pd
import numpy as np
import re
import math as m
from collections import Counter
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopWordList = stopwords.words('english')

inPath = 'cranfieldDocs'
outPath = 'preprocessed_cranfieldDocs'
query = 'queries.txt'
relevance = 'relevance.txt'
if not os.path.isdir(outPath):
    os.mkdir(outPath)
filenames = os.listdir(inPath)  

st = PorterStemmer()
shortword = re.compile(r'\W*\b\w{1,2}\b')


def tokenize(data):     # this function reduces the data passed into tokens

    lines = data.lower()
    lines = re.sub('[^A-Za-z]+', ' ', lines)
    tokens = word_tokenize(lines)
    cleanTokens = [word for word in tokens if word not in stopWordList]
    stemmwsTokens = [st.stem(word) for word in cleanTokens]
    stemmedTokens = [word for word in stemmwsTokens if word not in stopWordList]
    stemmedTokens = ' '.join(map(str, stemmedTokens))
    stemmedTokens = shortword.sub('',stemmedTokens)
    return stemmedTokens

def tokenExtraction(beautSoup, tag):   # Function to extract tokens from cranfield docs
    content = beautSoup.findAll(tag)
    content = ''.join(map(str, content))   
    content = content.replace(tag, '')
    content = tokenize(content)
    return content

for fname in filenames:         # Preprocessing all the documents in the cranfieldDocs directory
    infilepath = inPath + '/' + fname
    outfilepath = outPath + '/' + fname
    with open(infilepath) as infile:
        with open(outfilepath, 'w') as outfile:
            fileData = infile.read()
            soup = BeautifulSoup(fileData, features="html.parser" )
            title = tokenExtraction(soup, 'title')
            text = tokenExtraction(soup, 'text')
            outfile.write(title)
            outfile.write(" ")
            outfile.write(text)
        outfile.close()
    infile.close()


q = open(query)     # Preprocessing the queries.txt file
queries = []
text = q.readlines()
for line in text:
    query_tokens = tokenize(line)
    queries.append(query_tokens)
    

allDocuments = []

for fname in filenames:         # Generating a list of all preprocessed docs
    outfilepath = outPath + '/' + fname
    with open(outfilepath) as file:
        fileData = file.read()       
        allDocuments.append(fileData)


numDocs = len(allDocuments)
print (numDocs)


DF = {}

for i in range(numDocs):    # creating a dictionary with tokens as keys and their occurence in the corpus as the values
    tokens = allDocuments[i].split()
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:    
    DF[i] = len(DF[i])
print(DF)


vocab_size = len(DF)
print(vocab_size)


vocab = [term for term in DF]
print(vocab)

doc = 0
tf_idf = {}

for i in range(numDocs):        # creating dictionary to store tf-idf values for each term 
    
    tokens = allDocuments[i].split()    
    counter = Counter(tokens)
    wordCount = len(tokens)
    
    for token in np.unique(tokens):        
        tf = counter[token]/wordCount
        df = DF[token] if token in vocab else 0
        idf = np.log((numDocs+1)/(df+1))        
        tf_idf[doc, token] = tf*idf
    doc += 1

D = np.zeros((numDocs, vocab_size))

for i in tf_idf:
    ind = vocab.index(i[1])
    D[i[0]][ind] = tf_idf[i]

print(D)

def cosineSimilarity(k, query): #function to calculate cosine similarity of queries with docs
    tokens = query.split()     
    cosinesList = []    
    queryVector = np.zeros((len(vocab)))    
    counter = Counter(tokens)
    wordCount = len(tokens)
    for token in np.unique(tokens):        
        tf = counter[token]/wordCount
        df = DF[token] if token in vocab else 0
        idf = m.log((numDocs+1)/(df+1))
        try:
            ind = vocab.index(token)
            queryVector[ind] = tf*idf
        except:
            pass
    for d in D:
        sim = np.dot(queryVector, d)/(np.linalg.norm(queryVector)*np.linalg.norm(d)) 
        cosinesList.append(sim)        
    if k == 0:        
        out = np.array(cosinesList).argsort()[::-1]        
    else:           
        out = np.array(cosinesList).argsort()[-k:][::-1]    
    return out

def docsList(k):   # to generate retrieval output as a list of query id and document id pairs 
    cosSimList = []
    for i in range(len(queries)):
        cs = [i, cosineSimilarity(k, queries[i])]
        cosSimList.append(cs)        
    return cosSimList

print ('\n' + '\n' )
print (" below is the output of the retrieval as a list of (queryid, documentid) pairs")
print ('\n' + '\n')
df = pd.DataFrame(docsList(0))
print (df)
print ('\n' + '\n' + '\n')
with open('DocListbyCosineSimilarity.txt', 'a') as f:
    f.write(df.to_string(header = False, index = False))

colnames=['query', 'relevance']         # retrieving relevance values from relevance.txt
rel = pd.read_csv(relevance, delim_whitespace=True, names=colnames, header=None)
rel.head(10)
relevanceList = []

queryRelevance = [] 
for i in range(1,11):
    relevanceList = rel[rel['query']==i]['relevance'].to_list()
    queryRelevance.append(relevanceList)
   
top = [10, 50, 100, 500]

def calculateRecallandPrecision(k):  # Function to calculate precison and recall   
    recall = []
    precision = []
    for i in range(len(queries)):
        a = len([value for value in docsList(k)[i][1].tolist() if value in queryRelevance[i]])     
        b = len(queryRelevance[i])
        r = a / b
        p = a / k
        recall.append(r)
        precision.append(p)
    return recall, precision  

for t in top:       # To print the output for Task 2
    pr=0
    re=0
    print("Top {0} documents in the rank list".format(t))
    re, pr = calculateRecallandPrecision(t)
    for i in range(len(queries)):
        print("Query: {0} \t Precision: {1} \t Recall: {2}\n".format(i+1,pr[i],re[i]))
    print("Avg Precision: {0}\n".format(np.mean(pr)))
    print("Avg Recall: {0}\n\n".format(np.mean(re)))
    


