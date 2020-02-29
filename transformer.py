import numpy as np
import tensorflow.compat.v1 as tf
from config import DataConfig as dcfg, ModelConfig as mcfg, TrainingConfig as tcfg

tf.disable_eager_execution()

def genFixedPositionEmbedding(batcSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batcSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)

        embeddedPosition.append(x)

    return np.array(embeddedPosition, dtype='float32')

class Transformer(object):
    def __init__(self, wordEmbedding):
        self.inputx = tf.placeholder(tf.int32, [None, dcfg.sequenceLength], name='inputx')
        self.inputy = tf.placeholder(tf.int32, [None], name='inputy')

        self.dropoutKeepProb = tf.placeholder(tf.float32, name='dropoutKeepProb')
        self.embeddedPosition = tf.placeholder(tf.float32, [None, dcfg.sequenceLength, dcfg.sequenceLength], name='embeddedPosition')

        l2loss = tf.constant(0.0)

        with tf.name_scope('embedding'):
            #wordEmbedding = [[1.0,2.34,3.45,4.34,5.23],[1.3,2.2,3.4,4.5,5.1]]
            aa = tf.cast(wordEmbedding, dtype=tf.float32, name='word2vec')
            self.WV = tf.Variable(aa, name='wv')
            self.embedded = tf.nn.embedding_lookup(self.WV, self.inputx)
            self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope('transformer'):
            for i in range(mcfg.numBlocks):
                with tf.name_scope('transformer-{}'.format(i + 1)):
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputx, queries=self.embeddedWords, keys=self.embeddedWords)
                    self.embeddedWords = self._feedForward(multiHeadAtt, [mcfg.filters, mcfg.embeddingSize + dcfg.sequenceLength])

            outputs = tf.reshape(self.embeddedWords, [-1, dcfg.sequenceLength * (mcfg.embeddingSize + dcfg.sequenceLength)])

        outputsize = outputs.get_shape()[-1]

        with tf.name_scope('dropout'):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)

        with tf.name_scope('output'):
            outputW = tf.get_variable('outputW', shape=[outputsize, dcfg.numClasses], initializer=tf.random_uniform_initializer(0.0, 1.0))
            outputB = tf.Variable(tf.constant(0.1, shape=[dcfg.numClasses]), name='outputB')

            l2loss += tf.nn.l2_loss(outputW)
            l2loss += tf.nn.l2_loss(outputB)

            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name='logits')

            self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name='predictions')

        with tf.name_scope('loss'):
            lossses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.reshape(self.inputy, [-1, 1]), dtype=tf.float32))

            self.loss = tf.reduce_mean(lossses) + mcfg.l2RegLambda * l2loss


    def _layerNormalization(self, inputs, scope='layerNorm'):
        epsilon = mcfg.epsilon

        inputShape = inputs.get_shape()

        paramsShape = inputShape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gama = tf.Variable(tf.ones(paramsShape))

        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gama * normalized + beta

        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope='multiheadAttention'):
        numHead = mcfg.numHeads
        keepProb = mcfg.keepProp

        if numUnits is None:
            numUnits = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, numHead, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHead, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHead, axis=-1), axis=0)

        similary = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))

        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** .5)

        keyMasks = tf.tile(rawKeys, [numHead, 1])
        #keyMasks_1 = tf.expand_dims(keyMasks, 1)

        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings, scaledSimilary)

        weights = tf.nn.softmax(maskedSimilary)

        outputs = tf.matmul(weights, V_)

        outputs = tf.concat(tf.split(outputs, numHead, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProb)

        outputs += queries

        outputs = self._layerNormalization(outputs)

        return outputs

    def _feedForward(self, inputs, filters, scope='feedForward'):
        params = {'inputs':inputs, 'filters':filters[0], 'kernel_size':1,'activation':tf.nn.relu, 'use_bias':True}

        outputs = tf.layers.conv1d(**params)

        params = {'inputs': outputs, 'filters': filters[1], 'kernel_size': 1, 'activation': None, 'use_bias': True}

        outputs = tf.layers.conv1d(**params)

        outputs += inputs

        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded











