import logging
import gensim
from gensim.models import word2vec

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

handler = logging.FileHandler("./data/out/runtime-out.log")
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

logger.info('Starting...')
logger.error('Error...')


if __name__ == '__main__':
    path = './data/out/wordEmbedding.txt'
    sentences = word2vec.LineSentence(path)

    model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8, window=5, min_count=5)

    model.wv.save_word2vec_format('./data/out/word2vec.bin', binary=True)

    wordVec = gensim.models.KeyedVectors.load_word2vec_format('./data/out/word2vec.bin', binary=True)


    logger.info('Finished.......')
