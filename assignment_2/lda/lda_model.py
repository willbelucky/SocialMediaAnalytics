import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities
from itertools import chain
from assignment_2.data.data_reader import get_speeches
from assignment_2.data.stop_words import custom_stop_words
from stop_words import get_stop_words

""" DEMO """
speeches = get_speeches()
doc_set = speeches['script'].tolist()


# remove common words and tokenize
stoplist = set(custom_stop_words)
print(stoplist)
print(len(stoplist))
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in doc_set]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

# Create Dictionary.
id2word = corpora.Dictionary(texts)
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=24, alpha='auto', eval_every=5, passes=20)

# Prints the topics.
for top in lda.print_topics(num_topics=24, num_words=10):
    print(top)
"""
for i in lda.show_topics(formatted=False, num_topics=lda.num_topics, num_words=len(lda.id2word)):
    print(i)
"""
# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]

# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
scores = list(chain(*[[score for topic_id, score in topic]
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print(threshold)


cluster1 = [j for i, j in zip(lda_corpus, doc_set) if i[0][1] > threshold]
cluster2 = [j for i, j in zip(lda_corpus, doc_set) if i[1][1] > threshold]
cluster3 = [j for i, j in zip(lda_corpus, doc_set) if i[2][1] > threshold]

print(cluster1)
print(cluster2)
print(cluster3)

