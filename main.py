""" Script to extract key-phrases with unsupervised method. """

from keyphrex import KeyphraseExtractor, Error
import errno
import os

def extract_keyphrases(text_files, top_n_terms=10, max_words=3):
    """
    info

    Args:
        text_files: a list of txt files
        top_n_terms: number of top-ranking keyphrases to be returned
        max_words: maximum number of words in a keyphrase
    Returns:
         the top n candidate keyphrases with the highest tf*idf values
    """

    kp_ex = KeyphraseExtractor()

    # Check input files
    for file in text_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    # Check max_words
    if max_words < 1:
        raise Error('A keyword can be between 1 - n words')

    return kp_ex.score_keyphrases_tfidf(text_files, top_n_terms=top_n_terms, max_words=max_words)

def write_txt(list):
    """
    info

    Args:
        list: the list of top n candidate keyphrases
    Returns:
        outputs the scoring results to a txt file
    """
    with open('output.txt', 'w') as f:
        print('Keyphrase  score', file=f)
    with open('output.txt', 'a') as f:
        for term, score in list:
            print("{}: {:0.2f}".format(term, score), file=f)

if __name__ == '__main__':
    spath = os.path.dirname(__file__)
    tfs = ['script.txt', 'transcript_1.txt', 'transcript_2.txt', 'transcript_3.txt']
    text_files = [os.path.join(spath, 'scripts', tf) for tf in tfs]
    terms = extract_keyphrases(text_files)
    write_txt(terms)
    print(terms)
