from vocab import Vocab
import numpy as np
import itertools
import argparse
import cPickle
import os

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', default=1, type=int)
	parser.add_argument('--directory', default='/Users/mujq10/polysemy/data/train/', type=str)
	parser.add_argument('--normalize', default=0, type=int)
	parser.add_argument('--maxSenNum', default=5, type=int)
	parser.add_argument('--maxIter', default=1, type=int)
	parser.add_argument('--refresh', default=0, type=int)
	parser.add_argument('--iterNum', default=1, type=int)
	parser.add_argument('--pcaRank', default=3, type=int)
	parser.add_argument('--vecDim', default=300, type=int)
	parser.add_argument('--primary', default=0, type=int)
	parser.add_argument('--decoding', default='hard', type=str)
	parser.add_argument('--funcWordFile', default='/Users/mujq10/polysemy/data/train/function-word-list.txt', type=str)
	parser.add_argument('--polyListFile', default='/Users/mujq10/polysemy/data/train/poly-list.txt', type=str)
	args = parser.parse_args()

	#################################
	# parameters
	vecDim = args.vecDim
	debug = (args.debug == 1)
	funcWordFile = args.funcWordFile
	directory = args.directory
	maxSenNum = args.maxSenNum
	decoding = args.decoding
	pcaRank = args.pcaRank
	algoPath = directory + 'kGrassMean/senNum-%d-pcaRank-%d/' % (maxSenNum, pcaRank)
	outputPath = algoPath + 'labelCorpus/'
	polyListFile = args.polyListFile
	corpusPath = 'corpus/'
	corpusName = 'wikiCorpus.txt'
	vocabInputFile = 'vocab.txt'
	vecInputFile = 'vectors.bin'
	digitCorpusPath = algoPath + 'digitCorpus/'
	window = 10
	contextSize = 10000
	kmeansIterMax = 100
	##################################



	vocab = Vocab(vecDim, directory, corpusPath, corpusName, vocabInputFile, vecInputFile, debug, funcWordFile)
	polyList = open(polyListFile, 'r').read().split()
	vocab.computeSenseVecs(np.random.permutation((polyList)), digitCorpusPath, algoPath, '', 
					       pcaRank, window, contextSize, maxSenNum, kmeansIterMax)
	
	vocab.processCorpus(polyList, outputPath, corpusPath, corpusName, decoding)

