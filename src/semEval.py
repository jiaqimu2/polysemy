from vocab import Vocab
import numpy as np
import argparse
import cPickle
import os



if __name__ == "__main__":

	dirBase = {True:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/',
		       False:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/functional/post/'}

	semEvalFile = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SemEval-2010-WSI/wordList/'


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
	corpusDir = 'corpus/'
	corpusName = 'wikiCorpus.txt'
	vocabInputFile = 'vocab.txt'
	vecInputFile = 'vectors.bin'
	window = 10
	##################################

	vocab = Vocab(vecDim, directory, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, isNonFunctional)

	for pcaRank in [3,4,5]:
		for maxSenNum in [5, 2, 10]:
			
			count = 0
			vocab.setParams(pcaRank, window)
			vecDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/kGrassMean/senNum-%d-pcaRank-%d/vecs/' % (maxSenNum, pcaRank)
			fout = open('/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SemEval-2010-WSI/results/senNum-%d-pcaRank-%d.key' % (maxSenNum, pcaRank), 'w')
			for fname in os.listdir(semEvalFile):
				f = open(semEvalFile + fname, 'r')
				polysemy = fname.split('.')[0]
				if os.path.isfile(vecDir + polysemy + '.bin'):
					_, polyLabels, pcaVecs = cPickle.load(open(vecDir + polysemy + '.bin', 'rb'))
				else:
					# print polysemy, 'not exists'
					count += 1
					continue
				for raw in f.readlines():
					if not raw.startswith(fname):
						continue
					idx, raw = raw.rstrip().split('\t')
					_, label, _, ret = vocab.disambiguate(pcaVecs, raw)

					if label == -1:
						# print raw
						label = 0
					
					print >>fout, '%s %s %s' % (fname, idx, fname + polyLabels[label])

			fout.close()

			print 'pcaRank', pcaRank, 'maxSenNum', maxSenNum, 'missing', count


	vocab = Vocab(vecDim, directory, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, 0)

	for pcaRank in [3,4,5]:

		count = 0
		vocab.setParams(pcaRank, window)
		vecDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/ksvd/vecs/' 
		fout = open('/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SemEval-2010-WSI/results/global-pcaRank-%d.key' % (pcaRank), 'w')
		for fname in os.listdir(semEvalFile):
			f = open(semEvalFile + fname, 'r')
			polysemy = fname.split('.')[0]
			if os.path.isfile(vecDir + polysemy + '.bin'):
				_, polyLabels, pcaVecs = cPickle.load(open(vecDir + polysemy + '.bin', 'rb'))
			else:
				# print polysemy, 'not exists'
				count += 1
				continue
			for raw in f.readlines():
				if not raw.startswith(fname):
					continue
				idx, raw = raw.rstrip().split('\t')
				_, label, _, ret = vocab.disambiguate(pcaVecs, raw)
			
				print >>fout, '%s %s %s' % (fname, idx, fname + polyLabels[label])

		fout.close()

		print 'pcaRank', pcaRank, 'global', 'missing', count

