from vocab import Vocab
import numpy as np
import argparse
import cPickle
import os

CORPUS_WINDOW = 20
WORDLIST = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/wordlist'

if __name__ == '__main__':

	dirBase = {True:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/',
		       False:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/functional/post/'}


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
	parser.add_argument('--primary', default=0, type=int)
	parser.add_argument('--nonfunctional', default=1, type=int)
	args = parser.parse_args()

	#################################
	# parameters
	vecDim = args.vecDim
	debug = (args.debug == 1)
	adapt = (args.adapt == 1)
	isNonFunctional = (args.nonfunctional == 1)
	directory = dirBase[isNonFunctional]
	maxSenNum = args.maxSenNum
	pcaRank = args.pcaRank
	isNonFunctional = (args.nonfunctional == 1)
	fileDir = '/projects/csl/viswanath/data/public/preposition/preposition_instances/'
	algoDir = '/projects/csl/viswanath/data/public/kGrassmean/pca-%d-sen-%d/' % (pcaRank, maxSenNum)
	corpusDir = 'corpus/'
	corpusName = 'wikiCorpus.txt'
	vocabInputFile = 'vocab.txt'
	vecInputFile = 'vectors.bin'
	window = 5
	contextSize = 10000
	kmeansIterMax = 100
	##################################

	vocab = Vocab(vecDim, directory, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, isNonFunctional)
	os.system('mkdir -p %s' % algoDir + 'vecs/')
	print algoDir

	for fname in os.listdir(fileDir):
		print fname
		f = open(fileDir + fname, 'r')
		vecFile = algoDir + 'vecs/' + fname.split('.')[0] + '.bin'
		contexts = list()
		trueLabels = list()
		for raw in f.readlines():
			label, context = raw.split('\t')
			contexts.append(context)
			trueLabels.append(label)
		f.close()

		centVecs = vocab.computeSenseVecFromContexts(contexts, vecFile,
						     	  	  				 pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax)

		algoLabels = list()
		for i, context in enumerate(contexts):
			_, label, _, _ = vocab.disambiguate(centVecs, context)
			if label == -1:
				print 'dimensionality fault:', context
			algoLabels.append(label)
			print 'true label: %s \t predict label: %d' % (trueLabels[i], algoLabels[i])

		