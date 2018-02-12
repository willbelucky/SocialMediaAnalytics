# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 12.
"""
import os

import gensim
from gensim import corpora
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

from assignment_2.data.data_reader import get_speeches


def get_lda_model(doc_set, num_topics, num_words):
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = RegexpTokenizer(r'\w+').tokenize(raw)

        # remove stop words from tokens
        # stem tokens
        stemmed_tokens = [PorterStemmer().stem(i) for i in tokens if i not in get_stop_words('en')]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

    print(ldamodel.print_topics(num_topics=num_topics, num_words=num_words))
    print(ldamodel.get_topics())
    print(ldamodel.top_topics(processes=os.cpu_count() or 1))


if __name__ == '__main__':
    # create sample documents
    speeches = get_speeches()

    # compile sample documents into a list
    doc_set = speeches['script'].tolist()

    get_lda_model(doc_set, 10, 4)
