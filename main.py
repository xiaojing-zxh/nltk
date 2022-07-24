import gensim, logging, os
import matplotlib.ticker as mticker



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import nltk

corpus = nltk.corpus.brown.sents()

fname = 'brown_skipgram.model'
if os.path.exists(fname):
    # load the file if it has already been trained, to save repeating the slow training step below
    model = gensim.models.Word2Vec.load(fname)
else:
    # can take a few minutes, grab a cuppa
    model = gensim.models.Word2Vec(corpus, vector_size=100, min_count=5, workers=2, epochs=50)
    model.save(fname)

words = "sky man girl boy green blue".split()
print(words)
for w1 in words:
    for w2 in words:
        print(w1, w2, model.wv.similarity(w1, w2))
import numpy as np
import matplotlib.pyplot as plt

M = np.zeros((len(words), len(words)))
for i, w1 in enumerate(words):
    for j, w2 in enumerate(words):
        M[i, j] = model.wv.similarity(w1, w2)

plt.imshow(M, interpolation='nearest')
plt.colorbar()

ax = plt.gca()
a=5
b=20

ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels( words, rotation=45)
ax.set_yticks([0,1,2,3,4,5])
ax.set_yticklabels( words)
plt.savefig("final.png")
"""
vector = model.wv['computer']  # get numpy vector of a word
print(vector)
sims = model.wv.most_similar('computer', topn=10)
print(sims)
"""