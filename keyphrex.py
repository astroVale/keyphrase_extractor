import itertools
import nltk
import string
import math
import gensim

class KeyphraseExtractor(object):
    """Key-phrase Extractor class."""

    def read_txt(self, file):
        """
        info

        Args:
            file: file .txt to read
        Returns:
            doc: set the connection with txt file to a variable
        """
        with open(file) as text:
            doc = text.read()
        return doc


    def extract_candidate_chunks(self, text_string, max_words=3):
        """
        info

        Args:
            text_string: a txt file
            max_words: maximum number of words in a keyphrase
        Returns:
            candidates: the candidate keyphrases
        """

        # Any number of adjectives followed by noun(s) and (optionally) joined
        # by a preposition to any number of adjectives followed by any number of nouns
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'

        # Exclude candidates that are stop words or punctuation
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))

        # Make chunk using regular expression
        chunker = nltk.chunk.regexp.RegexpParser(grammar)

        # Tokenize and POS-tag
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text_string))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))

        # Join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word, pos, chunk in group).lower()
                      for key, group in itertools.groupby(all_chunks, lambda l: l[2] != 'O') if key]

        # Filter by maximum keyphrase length
        candidates = list(filter(lambda l: len(l.split()) <= max_words, candidates))

        candidates = [cand for cand in candidates
                      if cand not in stop_words and not all(char in punct for char in cand)]

        return candidates


    def score_keyphrases_tfidf(self, text_files, top_n_terms=10, max_words=3):
        """
        info

        Args:
            text_files: a list of txt files
            top_n_terms: number of top-ranking keyphrases to be returned
            max_words: maximum number of words in a keyphrase
        Returns:
            top_terms[:top_n_terms]: the top n candidate keyphrases with the highest tf*idf values
        """

        # Extract candidate chunks from each text in text_files
        chunked_texts = [self.extract_candidate_chunks(self.read_txt(text), max_words=max_words) for text in text_files]

        # Map id and term
        dictionary = gensim.corpora.Dictionary(chunked_texts)
        corpus = [dictionary.doc2bow(boc_text) for boc_text in chunked_texts]

        # tf*idf frequency model
        tfidf = gensim.models.TfidfModel(corpus[0:], normalize=False,
                                         wglobal=lambda df, D: math.log((1 + D) / (1 + df)) + 1)
        corpus_tfidf = tfidf[corpus][0]

        # Sort by score
        sorted_corpus = sorted(corpus_tfidf, key=lambda item: item[1], reverse=True)

        # Compute top n terms
        top_terms = [(dictionary[s[0]], s[1]) for s in sorted_corpus]

        return top_terms[:top_n_terms]


class Error(Exception):
   """Raised when the max words value is < 1."""

   def __init__(self, expr):
       self.expr = expr

   def __str__(self):
       return (repr(self.expr))
