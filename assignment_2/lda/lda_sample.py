# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 12.
"""
import re
import gensim
from gensim import corpora
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

from assignment_2.data.data_reader import get_speeches
from assignment_2.data.stop_words import custom_stop_words


def get_lda_model(doc_set, num_topics, num_words):
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = RegexpTokenizer(r'[a-z]+').tokenize(raw)

        # remove general stop words from tokens
        not_stop_words = [i for i in tokens if i not in get_stop_words('en') and i not in custom_stop_words]

        # remove custom stop words from tokens
        stemmed_tokens = [PorterStemmer().stem(i) for i in not_stop_words]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

    topics = ldamodel.print_topics(num_topics=num_topics, num_words=num_words)
    r = re.compile('[a-z]+')
    for index, topic in enumerate(topics):
        words = r.findall(topic[1])
        print(index, words)

    return ldamodel


if __name__ == '__main__':
    from assignment_2.data.president import *

    # create sample documents
    speeches = get_speeches()

    # compile sample documents into a list
    doc_set = speeches['script'].tolist()

    # Print 24 topics of all speeches.
    get_lda_model(doc_set, 24, 20)

    # Print 1 topic for every speech.
    # for i in doc_set:
        # get_lda_model([i], 1, 20)

    # Print data and president list.
    # for ((date, president), script) in speeches.iterrows():
    #     print(date, president)