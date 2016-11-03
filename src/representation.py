from vocab import Vocab
import numpy as np
import itertools
import argparse
import cPickle
import os

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CORPUS_WINDOW = 20


def generateSentences(targetWord, corpusDir):

	contexts = list()
	for fdir in os.listdir(corpusDir):
		print 'handling %s' % (fdir, )
		for fname in os.listdir(corpusDir + fdir):
			
			fin = open('%s%s/%s' % (corpusDir, fdir, fname), 'r')
			for doc in fin.readlines():
				doc = doc.rstrip().lower().split()
				texts = list()
				labels = list()
				for word in doc:
					if '|*|*|' not in word:
						label, text = '', word
					else:
						label, text = word.split('|*|*|')
					texts.append(text)
					labels.append(label)
				for i, text in enumerate(texts):
					if text != targetWord:
						continue
					contexts.append(' '.join(texts[max(0, i - CORPUS_WINDOW): i] + ['<b>', text, '</b>'] + texts[i+1: min(i + CORPUS_WINDOW, len(doc))]))
			fin.close()
			if debug:
				break
	return contexts


if __name__ == '__main__':

	dirBase = {True:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/',
		       False:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/functional/post/'}

	wordListBase = {'princeton': '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/princeton/wordlist',
					'scws': '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SCWS/wordList'}


	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', default=1, type=int)
	parser.add_argument('--normalize', default=0, type=int)
	parser.add_argument('--maxSenNum', default=5, type=int)
	parser.add_argument('--maxIter', default=1, type=int)
	parser.add_argument('--refresh', default=0, type=int)
	parser.add_argument('--iterNum', default=1, type=int)
	parser.add_argument('--adapt', default=0, type=int)
	parser.add_argument('--pcaRank', default=3, type=int)
	parser.add_argument('--vecDim', default=300, type=int)
	parser.add_argument('--nonfunctional', default=1, type=int)
	parser.add_argument('--task', default='scws', type=str)
	parser.add_argument('--decoding', default='hard', type=str)
	args = parser.parse_args()

	#################################
	# parameters
	task = args.task
	decoding = args.decoding
	vecDim = args.vecDim
	debug = (args.debug == 1)
	adapt = (args.adapt == 1)
	isNonFunctional = (args.nonfunctional == 1)
	directory = dirBase[isNonFunctional]
	maxSenNum = args.maxSenNum
	pcaRank = args.pcaRank
	algoDir = directory + 'kGrassMean/senNum-%d-pcaRank-%d/' % (maxSenNum, pcaRank)
	isNonFunctional = (args.nonfunctional == 1)
	corpusDir = 'corpus/'
	corpusName = 'wikiCorpus.txt'
	vocabInputFile = 'vocab.txt'
	vecInputFile = 'vectors.bin'
	digitCorpusDir = algoDir + 'digitCorpus/'
	window = 10
	contextSize = 10000
	kmeansIterMax = 100
	##################################


	vocab = Vocab(vecDim, directory, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, isNonFunctional)
	polyList = open(wordListBase[args.task], 'r').read().split()

	vocab.computeSenseVecs(np.random.permutation((polyList)), digitCorpusDir, algoDir, '', 
					       pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax)

	outDir = '%s../%s/%s/senNum-%d-pcaRank-%d/' % (directory, task, decoding, maxSenNum, pcaRank)
	print outDir
	vocab.processCorpus(polyList, outDir, corpusDir, corpusName, decoding)

