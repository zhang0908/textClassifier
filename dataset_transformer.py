import pandas as pd
from config import DataConfig as dcfg, TrainingConfig as tcfg, ModelConfig as mcfg
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
import json
import tensorflow as tf

class Dataset_transformer(object):
    def __init__(self):

        self.wordEmbedding = None
        self.lable2index = None

        self.trainReviews = None
        self.trainLables = None
        self.validReviews = None
        self.validLables = None


    def _readData(self, filePath):
        df = pd.read_csv(filePath)

        lables = df['sentiment'].tolist()

        reviews = df['review']
        reviews = [review.strip().split(' ') for review in reviews]

        return reviews, lables

    def _lable2index(self, lables, label2index):
        return [label2index[lable] for lable in lables]

    def _word2index(self, reviews, word2index):
        return [[word2index.get(word, word2index['UNK']) for word in review] for review in reviews]

    def _genTrainEvalData(self, x, y, word2index, rate):
        reviews = []

        for review in x:
            if len(review) > dcfg.sequenceLength:
                reviews.append(review[:dcfg.sequenceLength])
            else:
                reviews.append(review + [word2index['PAD']] * (dcfg.sequenceLength - len(review)))

        trainIndex = int(len(reviews) * rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLables = np.asarray(y[:trainIndex], dtype="float32")

        validReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        validLables = np.asarray(y[trainIndex:], dtype="float32")

        return trainReviews, trainLables, validReviews, validLables



    def _getWordEmbedding(self, words):
        wordVec = KeyedVectors.load_word2vec_format('./data/out/word2vec.bin', binary=True)
        vocab = []
        wordEmbedding = []

        vocab.append('PAD')
        vocab.append('UNK')
        wordEmbedding.append(np.zeros(mcfg.embeddingSize, dtype=np.float32))
        wordEmbedding.append(np.random.randn(mcfg.embeddingSize))

        for word in words:
            try:
                wordEmbedding.append(wordVec.wv[word])
                vocab.append(word)
            except:
                print(word + '不存在于词向量中')
                #wordEmbedding.append(wordVec.wv['UNK'])

        return vocab, np.array(wordEmbedding)

    def _genVocabulary(self, reviews, lables):
        allWords = [word for review in reviews for word in review]

        wordCount = Counter(allWords)
        sortedWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        words = [item[0] for item in sortedWordCount if item[1] > 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)

        self.wordEmbedding = wordEmbedding

        word2index = dict(zip(vocab, list(range(len(vocab)))))

        uniqueLable = list(set(lables))
        lable2index = dict(zip(uniqueLable, list(range(len(uniqueLable)))))

        self.lable2index = lable2index

        with open('./data/out/word2index.json', 'w', encoding='utf-8') as f:
            json.dump(word2index, f)

        with open('./data/out/lable2index.json', 'w', encoding='utf-8') as f:
            json.dump(lable2index, f)

        return word2index, lable2index

    def dataGen(self):
        reviews, lables = self._readData(dcfg.dataSource)

        word2index, lable2index = self._genVocabulary(reviews, lables)

        lableIdxs = self._lable2index(lables, lable2index)
        reviewIdxs = self._word2index(reviews, word2index)

        trainReviews, trainLables, validReviews, validLables = self._genTrainEvalData(reviewIdxs, lableIdxs, word2index, dcfg.rate)

        self.trainReviews = trainReviews
        self.trainLables = trainLables
        self.validReviews = validReviews
        self.validLables = validLables


def nextBatchData(x, y, batchSize):
    perm = np.arange(len(x))
    np.random.shuffle(perm)

    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchx = np.array(x[start:end], dtype='int64')
        batchy = np.array(y[start:end], dtype='float32')

        yield batchx, batchy


data = Dataset_transformer()
data.dataGen()
