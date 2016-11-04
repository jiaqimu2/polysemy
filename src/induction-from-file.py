from vocab import Vocab
import numpy as np
import argparse
import cPickle
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', default=1, type=int)
	parser.add_argument('--directory', default='/Users/mujq10/polysemy/data/train/', type=str)
	parser.add_argument('--normalize', default=0, type=int)
	parser.add_argument('--maxSenNum', default=5, type=int)
	parser.add_argument('--maxIter', default=1, type=int)
	parser.add_argument('--refresh', default=0, type=int)
	parser.add_argument('--iterNum', default=1, type=int)
	parser.add_argument('--adapt', default=0, type=int)
	parser.add_argument('--pcaRank', default=3, type=int)
	parser.add_argument('--vecDim', default=300, type=int)
	parser.add_argument('--primary', default=0, type=int)
	parser.add_argument('--filePath', default='/Users/mujq10/polysemy/data/train/contexts/', type=str)
	parser.add_argument('--funcWordFile', default='/Users/mujq10/polysemy/data/train/function-word-list.txt', type=str)
	args = parser.parse_args()

	#################################
	# parameters
	vecDim = args.vecDim
	debug = (args.debug == 1)
	adapt = (args.adapt == 1)
	funcWordFile = args.funcWordFile
	directory = args.directory
	maxSenNum = args.maxSenNum
	pcaRank = args.pcaRank
	algoPath = directory + 'kGrassMean/senNum-%d-pcaRank-%d/' % (maxSenNum, pcaRank)
	filePath = args.filePath
	corpusPath = 'corpus/'
	corpusName = 'wikiCorpus.txt'
	vocabInputFile = 'vocab.txt'
	vecInputFile = 'vectors.bin'
	window = 5
	contextSize = 10000
	kmeansIterMax = 100
	##################################

	vocab = Vocab(vecDim, directory, corpusPath, corpusName, vocabInputFile, vecInputFile, debug, funcWordFile)
	os.system('mkdir -p %s' % algoPath + 'vecs/')

	for fname in os.listdir(filePath):
		print fname
		f = open(filePath + fname, 'r')
		vecFile = algoPath + 'vecs/' + fname.split('.')[0] + '.bin'
		contexts = list()
		for raw in f.readlines():
			context = raw.rstrip()
			contexts.append(context)
		f.close()

		centVecs = vocab.computeSenseVecFromContexts(contexts, fname.split('.')[0], vecFile,
						     	  	  				 pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax)

		algoLabels = list()
		for i, context in enumerate(contexts):
			_, label, _, _ = vocab.disambiguate(centVecs, context)
			algoLabels.append(label)
			