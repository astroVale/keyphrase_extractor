# Keyphrases Extraction: keyphrex
Python 3 module which generates the most important key-phrase (or key-words) from a txt document based on a corpus of texts.

Based on the heuristic method described in [this blog](http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/), it extracts candidate keyphrases from the given document(s), it then generates a score for the candidate keyphrases based on term-frequency inverse-document-frequency analysis ([tf-idf](https://en.wikipedia.org/wiki/Tf–idf)) and finally returns a text file with the *n* highest *tf-idf* values.
This is a very simple unsupervised approach to extract candidate keyphrases from a document. There are many other methods and algorithms which could be applied and amongst them supervised approaches like [KEA](http://www.nzdl.org/Kea/description.html) or Linear Ranking SVM would achieve better performance, given that [good training data](https://github.com/snkim/AutomaticKeyphraseExtraction) is provided.   

## Scripts:
* ```keyphrex.py``` - Python class to extract key-phrases from text files
* ```main.py``` - The main script to run and get key-phrases
*  ```keyphrex.ipynb``` - Jupyter Notebook for keywords extraction for a quick glance
* The *scripts.zip* archive contains:
   - one script file (*script.txt*)
   - 3 transcript files (*transcript1...3.txt*)
   
## Instructions
1. Install the necessary requirements using the ```requirements.txt``` file:
```
pip install -r requirements.txt
```
2. Install the needed nltk packages in Python:
```
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```  
3. Unzip the archive *scripts.zip*
4. Run the ```main.py``` script

## Usage

The function ```extract_keyphrases(text_files, top_n_terms=10, max_words=3)``` takes a list of txt files as input, computes the most important keywords (a keyword can be between 1 - n words) and calculates the document frequencies for the transcripts.

Optional parameters:
* ```top_n_terms```: the number of top-ranking keyphrases (keywords) to be returned (by default is `10`)
* ```max_words```: the desired maximum number of words in a keyphrase (by default is `3`)