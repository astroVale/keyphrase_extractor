""" Script to extract key-phrases with unsupervised method. """

from keyphrases_extraction import KeyphraseExtractor
import errno
import os

def extract_keyphrases(text_files, top_n_terms=10, max_words=3):
    # Check input files
    for file in text_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    kp_ex = KeyphraseExtractor()
    return kp_ex.score_keyphrases_tfidf(text_files, top_n_terms=top_n_terms, max_words=max_words)

if __name__ == '__main__':
    spath = os.path.dirname(__file__)
    tfs = ['script.txt', 'transcript_1.txt', 'transcript_2.txt', 'transcript_3.txt']
    text_files = [os.path.join(spath, tf) for tf in tfs]
    terms = extract_keyphrases(text_files)
    kp_ex.write_txt(terms)
    print(terms)
