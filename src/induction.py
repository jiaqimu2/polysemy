from vocab import Vocab
import numpy as np
import itertools
import argparse
import cPickle
import os

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def init_plotting():
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['font.size'] = 10
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'upper left'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['figure.autolayout'] = True

    # plt.gca().spines['right'].set_color('none')
    # plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.tight_layout()

init_plotting()

CORPUS_WINDOW = 20

def generateSentences(targetWord, corpusPath):

	contexts = list()
	for fdir in os.listdir(corpusPath):
		print 'handling %s' % (fdir, )
		for fname in os.listdir(corpusPath + fdir):
			
			fin = open('%s%s/%s' % (corpusPath, fdir, fname), 'r')
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
	parser.add_argument('--funcWordFile', default='/Users/mujq10/polysemy/data/train/function-word-list.txt', type=str)
	parser.add_argument('--polyListFile', default='/Users/mujq10/polysemy/data/train/poly-list.txt', type=str)
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
					       pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax)


	## generate sentences
	for poly in np.random.permutation(polyList):

		textFile = algoPath + 'texts/%s.txt' % poly
		vecFile = algoPath + 'vecs/%s.bin' % poly
		figFile = algoPath + 'figures/%s-cossim.pdf' % poly
		if (not os.path.isfile(vecFile)):
			continue

		print 'working on %s' % poly

		_, polySet, centVec = cPickle.load(open(vecFile, 'rb'))
		if not os.path.isfile(textFile):
			contexts = generateSentences(poly, directory + corpusPath)
			retContexts = list()
			errs = list()
			labels = list()
			for context in contexts:
				err, label, ratio, retStr = vocab.disambiguate(centVec, context)
				errs.append(err)
				labels.append(label)
				retContexts.append(retStr)

			retContexts = np.array(retContexts)
			labels = np.array(labels)
			errs = np.array(errs)

			print labels
			print len(labels)

			f = open(textFile, 'w')
			for label in set(labels):
				labelIdx = np.where(labels == label)[0]
				sortIdx = np.argsort(errs[labelIdx])
				count = 0
				for idx in sortIdx:
					count += 1
					print >>f, "%s\t%0.6f\t%s" % (polySet[label], errs[labelIdx[idx]], retContexts[labelIdx[idx]])
				print >>f, "\n"*10
			f.close()
		if not os.path.isfile(figFile):
			plt.figure()
			cm = np.abs(np.dot(centVec, centVec.T))
			plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
			plt.colorbar()
			plt.clim(0, 1)

			thresh = cm.max() / 2.
			for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, '%0.2f' % cm[i, j],
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black")

			plt.savefig(figFile)
			plt.close()
