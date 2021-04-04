To Compile & Run: Make sure that files ‘assignment2.py’, ‘queries.txt’, ‘relevance.txt’ and the ‘Cranfield’ document collection (cranfieldDocs folder from cranfield.tar) extracted from the cranfield.tar file are in the same folder.

• Navigate to the directory containing the above files, open terminal and run the command python3 assignment2.py (all the code is written and executed in python 3.7)
Note: Make sure that the packages such as OS, re, math and ‘nltk’, ‘BeautifulSoup’ pandas and numpy are installed before running.

• Function ‘tokenize’ is used to preprocess the string given as input. It converts the passed data to lower case, removes the punctuations and numbers, splits on whitespaces, removes stop words, performs stemming & removes words with one- or two-characters length.
• Function ‘tokenExtraction’ is used to extract tokens from the text between a specific SGML tag.
• Function ‘cosineSimilarity’ is used to calculate cosine similarity between query and documents
• Function ‘docsList’ is created to generate retrieval output as a list of query id and document id pairs
• Function ‘calculateRecallandPrecision’ calculates recall and precision for each query and generates a list for every query.

Note: For the output of the retrieval as a list of (query id, document id) pairs, refer to the DocListbyCosineSimilarity.txt file generated after the execution of the program in the same directory as the program file