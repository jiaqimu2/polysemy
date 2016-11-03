from vocab import Vocab
import numpy as np
import argparse
import cPickle
import os

def cosSim(array1, array2):

	if (np.linalg.norm(array1) * np.linalg.norm(array2)) == 0:
		return 0

	if False: #'global' in outFile:
		return np.abs(np.dot(array1, array2)) / (np.linalg.norm(array1) * np.linalg.norm(array2))
	else:
		return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

def taskSCWS(vocab, centVecs, repVecs, outFile, threshold):

	fout = open(outFile, 'w')
	fin = open(scwsFile, 'r')

	for raw in fin.readlines():
		raw = raw.split('\t')
		poly1 = raw[1].lower()
		poly1Context = raw[5].lower()
		poly2 = raw[3].lower()
		poly2Context = raw[6].lower()
		humanScore = float(raw[7])

		try:
			centVec1 = centVecs[poly1]
			centVec2 = centVecs[poly2]
			repVec1 = repVecs[poly1]
			repVec2 = repVecs[poly2]
		except:
			print >>fout, -1, -1
			continue

		err1, label1, _, _ = vocab.disambiguate(centVec1, poly1Context)
		err2, label2, _, _ = vocab.disambiguate(centVec2, poly2Context)

		if err1 > threshold and decoding == 'hard':
			label1 = 'IDK'
		if err2 > threshold and decoding == 'hard':
			label2 = 'IDK'

		try:
			hardSim = cosSim(repVec1[label1], repVec2[label2])
		except:
			print >>fout, -1, -1
			continue

		prob1, _, _ = vocab.disambiguate_soft(centVec1, poly1Context)
		prob2, _, _ = vocab.disambiguate_soft(centVec2, poly2Context)

		softSim = 0
		prob = 0
		for i1, p1 in enumerate(prob1):
			for i2, p2 in enumerate(prob2):
				try:
					softSim += p1 * p2 * cosSim(repVec1[i1], repVec2[i2])
					prob += p1 * p2
				except:
					continue

		softSim /= prob

		print >>fout, softSim, hardSim
		

if __name__ == "__main__":

	dirBase = {True:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/',
		       False:	'/projects/csl/viswanath/data/jiaqimu2/wordnet/data/functional/post/'}

	scwsFile = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SCWS/ratings.txt'
	outDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SCWS/results/'
	wordlist = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/SCWS/wordList'


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
	polyVocab = open(wordlist, 'r').read().split()

	os.system('mkdir -p /projects/csl/viswanath/data/jiaqimu2/wordnet/data/SCWS/results/')

	for pcaRank in [3]:

		vocab.setParams(pcaRank, window)
		for decoding in ['soft', 'hard']:
			for maxSenNum in [5, 2, 10]:

				vecDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/kGrassMean/senNum-%d-pcaRank-%d/vecs/' % (maxSenNum, pcaRank)
				vocabDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/scws/%s/senNum-%d-pcaRank-%d/' % (decoding, maxSenNum, pcaRank)

				centVecs = dict()
				repVecs = dict()

				if not (os.path.isfile(vocabDir + vocabInputFile) and os.path.isfile(vocabDir + vecInputFile)):
					continue

				newVocab = Vocab(vecDim, vocabDir, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, isNonFunctional)
				
				for word in polyVocab:
					if os.path.isfile(vecDir + word + '.bin'):
						_, polyLabels, pcaVec = cPickle.load(open(vecDir + word + '.bin', 'rb'))
						repVec = dict()
						for i, label in enumerate(polyLabels):
							try:
								if label != '.0':
									try:
										repVec[i] = newVocab.vecMatrixNorm[newVocab.vocabIndex[word + label]]
									except:
										repVec[i] = newVocab.vecMatrixNorm[newVocab.vocabIndex[word + '.0']]
								else:
									repVec['IDK'] = newVocab.vecMatrixNorm[newVocab.vocabIndex[word + label]] 
							except:
								continue
						if len(repVec) == 0:
							continue
						centVecs[word] = pcaVec
						repVecs[word] = repVec
					else:
						continue

				outFile = outDir + '%s-rank-%d-senNum-%d' % (decoding, pcaRank, maxSenNum)
				taskSCWS(vocab, centVecs, repVecs, outFile, 0.6)

		# GloVe
		centVecs = dict()
		repVecs = dict()
		vecDir = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/ksvd/vecs/'
		for word in polyVocab:
			if os.path.isfile(vecDir + word + '.bin'):
				_, _, vec = cPickle.load(open(vecDir + word + '.bin', 'rb'))
				centVecs[word] = vec
				repVecs[word] = vec

		outFile =  outDir + 'global-%s-rank-%d' % (decoding, pcaRank)
		taskSCWS(vocab, centVecs, repVecs, outFile, 1)

	# word2vec
	repVecs = dict()
	centVecs = dict()
	for word in polyVocab:
		try:
			centVecs[word] = np.array([vocab.vecMatrixNorm[vocab.vocabIndex[word]]])
			repVecs[word] = np.array([vocab.vecMatrixNorm[vocab.vocabIndex[word]]])
		except:
			continue

	outFile = outDir + 'word2vec'
	taskSCWS(vocab, centVecs, repVecs, outFile, 1)




				