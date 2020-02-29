import pandas as pd
from bs4 import BeautifulSoup

class DataSet(object):

    def __init__(self, unlabledTrainDataPath, labledTrainDataPath):
        self.unlabledTrainDataPath = unlabledTrainDataPath
        self.labledTrainDataPath = labledTrainDataPath

    def doWithOneReview(self, review):
        beau = BeautifulSoup(review)
        review = beau.get_text()
        review = review.replace('\\', ' ').replace('\'', ' ').replace('/', ' ').replace('"', ' ').replace(',', ' ')\
            .replace('.', ' ').replace('?', ' ').replace('(', ' ').replace(')', ' ')
        review = review.strip().split(' ')
        review = [word for word in review if len(word) > 0]
        review = ' '.join(review)
        return review


    def process(self, outPath):
        with open(self.unlabledTrainDataPath, 'r', encoding='utf-8', errors='ignore') as f:
            unlabeldTrainData = [line.strip().split('\t') for line in f.readlines() if len(line.strip().split('\t')) == 2]

        with open(self.labledTrainDataPath, 'r', encoding='utf-8', errors='ignore') as f:
            labledTrainData = [line.strip().split('\t') for line in f.readlines() if len(line.strip().split('\t')) == 3]

        unlabledTrainDF = pd.DataFrame(unlabeldTrainData[1:], columns=unlabeldTrainData[0])
        labledTrainDF = pd.DataFrame(labledTrainData[1:], columns=labledTrainData[0])

        unlabledTrainDF['review'] = unlabledTrainDF['review'].apply(self.doWithOneReview)
        labledTrainDF['review'] = labledTrainDF['review'].apply(self.doWithOneReview)

        allReview = pd.concat([unlabledTrainDF['review'], labledTrainDF['review']], axis=0)

        allReview.to_csv(outPath, index=False)

if __name__ == '__main__':
    unlabledTrainDataPath = './data/rawData/unlabeledTrainData.tsv'
    labledTrainDataPath = './data/rawData/labeledTrainData.tsv'
    outPath = './data/out/wordEmbedding.txt'
    dataset = DataSet(unlabledTrainDataPath=unlabledTrainDataPath, labledTrainDataPath=labledTrainDataPath)
    dataset.process(outPath)
    print('Finished......')