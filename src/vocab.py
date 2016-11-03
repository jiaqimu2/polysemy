from nltk.corpus import wordnet as wn
from collections import defaultdict
from os.path import isfile
# from sklearn.decomposition import PCA
from pca import PCA
from scipy.io import savemat
from scipy.stats import spearmanr
from scipy.linalg import orth
import matplotlib 
matplotlib.use('Agg')
from numpy.linalg import norm
from math import sqrt
from multiprocessing import cpu_count
import multiprocessing
import matplotlib.pyplot as plt
import cPickle
import threading
import random
import Queue
import argparse
import scipy 
import numpy as np
import sys
import array
import numpy.random as rn
import cPickle as pickle
import itertools
import os
import logging
import time
import struct
import io

logging.basicConfig(level=logging.INFO,
					stream=sys.stdout,
                    format="%(levelname)s: %(process)s - %(message)s",
                    )

WORD2VEC_SCRIPT = '/projects/csl/viswanath/data/jiaqimu2/wordnet/src/word2vec/word2vec-init.sh'
FUNCWORD = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/funcWords.txt'
## $1 wikicorpus
## $2 vectors.bin
## $3 vocab.txt 
## $4 dim
## $5 min_count


def cosSim(array1, array2):

	if (np.linalg.norm(array1) * np.linalg.norm(array2)) == 0:
		return 0

	return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

def l2Err(array1, array2):
	return np.linalg.norm(array1-array2)/np.linalg.norm(array1)

def normalizeMatrix(mat):
	matNorm = np.zeros(mat.shape)
	d = (np.sum(mat ** 2, 1) ** (0.5))
	matNorm = (mat.T / d).T

	return matNorm

def npArray():
	return np.zeros((vecDim))

def dictInt():
	return defaultdict(int)

def cosSimSubspace(array1, arrayList):

	vec = normalizeMatrix(np.array(arrayList))
	v = orth(vec.T)
		
	coeffs = np.dot(v.T, array1.T)
	d = np.sum(coeffs ** 2) ** 0.5
	return d/np.linalg.norm(array1)

class Vocab:

	def __init__(self, vecDim, directory, corpusDir, corpusName,
		               vocabInputFile, vectorInputFile, debug, isFunctional):

		self.vecDim = vecDim
		self.corpusDir = directory + corpusDir
		self.corpusName = directory + corpusName
		self.vocabFile = directory + vocabInputFile
		self.vecFile = directory + vectorInputFile
		self.debug = debug

		# if not (os.path.isfile(self.vocabFile) and os.path.isfile(self.vecFile)):
		# 	os.system('rm -r %s' % self.corpusName)
		# 	os.system('for file in %s*/*; do; cat $file >> %s; done' % (self.corpusDir, self.corpusName))
		# 	if debug:
		# 		os.system('%s %s %s %s %d 1000' % (WORD2VEC_SCRIPT, self.corpusName, self.vecFile, self.vocabFile, self.vecDim))
		# 	else:
		# 		os.system('%s %s %s %s %d 100' % (WORD2VEC_SCRIPT, self.corpusName, self.vecFile, self.vocabFile, self.vecDim))


		self.readVocabFromFile()
		self.readVectorFromFile()
		self.readFuncWords(isFunctional)

		if debug:
			self.sem = multiprocessing.BoundedSemaphore(2)
		else:
			self.sem = multiprocessing.BoundedSemaphore(multiprocessing.cpu_count() - 1)


	def setParams(self, pcaRank, window):
		self.pcaRank = pcaRank
		self.window = window

	def readFuncWords(self, isFunctional):
		funcWords = set()
		if isFunctional:
			f = open(FUNCWORD, 'r')
			for line in f.readlines():
				funcWords.add(line.rstrip())

		self.funcWords = funcWords

	def readVocabFromFile(self):

		vocabList = list()
		vocabIndex = dict()
		vocabCount = list()

		f = open(self.vocabFile, "r")
		idx = 0
		for line in f.readlines():
			raw = line.lower().split()
			vocabList.append(raw[0])
			vocabCount.append(int(raw[1]))
			vocabIndex[raw[0]] = idx 
			idx += 1
			if idx == 200000:
				break

		self.vocabList = vocabList
		self.vocabIndex = vocabIndex
		self.vocabSize = len(self.vocabList)
		self.vocabCount = vocabCount

		print >>sys.stdout, "Done loading vocabulary."

	def readVectorFromFile(self):

		vecDim = self.vecDim
		vecMatrix = array.array('f')
		vecMatrix.fromfile(open(self.vecFile, 'rb'), self.vocabSize * vecDim)
		vecMatrix = np.reshape(vecMatrix, (self.vocabSize, vecDim))[:, 0:vecDim]

		vecMatrixNorm = normalizeMatrix(vecMatrix)
		self.vecMatrixNorm = vecMatrixNorm
		self.vecMatrix = vecMatrix

		print >>sys.stdout, "Done loading vectors."


	###################################################
	## induction

	def computeSenseVecFromContexts(self, contexts, vecFile, 
								    pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax):

		################
		# parameters
		self.pcaRank = pcaRank
		self.window = window
		self.contextSize = contextSize
		self.adapt = adapt
		self.maxSenNum = maxSenNum
		self.kmeansIterMax = kmeansIterMax
		#####################

		contextList = self.getContextIdList(contexts)
		
		vec = self.kMeansSubspace(contextList)
		print vecFile
		cPickle.dump(vec, open(vecFile, 'wb'))

		return vec

	def getContextIdList(self, contexts):

		contextList = list()
		for context in contexts:
			context = context.lower().rstrip().split()
			try:
				start = context.index('<b>')
				end = context.index('</b>')
			except:
				print context
				continue
			contextId = list()
			idx = start
			count = 0
			while idx > 0 and count < self.window:
				idx -= 1
				if context[idx] in self.funcWords:
					continue
				try:
					contextId.append(self.vocabIndex[context[idx]])
					count += 1
				except:
					pass
			idx = end
			count = 0
			while idx < len(context) - 1 and count < self.window:
				idx += 1
				if context[idx] in self.funcWords:
					continue
				try:
					contextId.append(self.vocabIndex[context[idx]])
					count += 1
				except:
					pass
			if len(contextId) < self.pcaRank:
				continue
			contextList.append(contextId)
		return contextList


	def computeSenseVecs(self, polyList, digitCorpusDir, algoDir, preDir, \
					       	   pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax):

		self.digitalizeCorpus(digitCorpusDir)

		################
		# parameters
		self.pcaRank = pcaRank
		self.window = window
		self.contextSize = contextSize
		self.adapt = adapt
		self.maxSenNum = maxSenNum
		self.kmeansIterMax = kmeansIterMax
		self.algoDir = algoDir
		self.preDir = preDir

		os.system('mkdir -p %s' % algoDir + 'vecs/')
		os.system('mkdir -p %s' % algoDir + 'figures/')
		os.system('mkdir -p %s' % algoDir + 'texts/')
		#####################


		randPolyVocab = np.random.permutation(polyList)
		processes = []
		for polysemy in randPolyVocab:
			self.sem.acquire()
			t = multiprocessing.Process(target=self.findContexts, args=(polysemy,))
			processes.append(t)
			t.start()

		for t in processes:
			t.join()

		print >>sys.stdout, 'Done computing senses.'

	def digitalizeCorpus(self, digitCorpus):

		os.system('mkdir -p ' + digitCorpus)
		self.digitCorpus = digitCorpus
		start = time.clock()
		count = 0
		corpus = self.corpusDir

		for fdir in os.listdir(corpus):

			os.system('mkdir -p '+ digitCorpus + fdir + '/')
			print >>sys.stdout, 'Digitalizing corpus from directory', fdir

			for fname in os.listdir(corpus+fdir):

				if isfile(digitCorpus + fdir + '/' + fname):
					continue

				f = open(corpus + fdir + '/' + fname, 'r')

				doc = f.read().rstrip().split()
				f.close()

				count += 1

				docIndex = list()
				
				for word in doc:
					if word in self.funcWords:
						continue
					try:
						docIndex.append(self.vocabIndex[word])
					except:
						pass
				docIndex.insert(0, len(docIndex))

				f = open(digitCorpus + fdir + '/' + fname, 'wb')
				f.write(struct.pack(str(len(docIndex))+'i', *docIndex))
				f.close()

				del docIndex

		end = time.clock()
		print >>sys.stdout, "Done preprocess corpus: ", end - start

	def findWord(self, word, polySet):
		'''
		input: word
		output: list of context indices
				list of context sentences
		'''

		window = self.window

		contextList = defaultdict(list)
		wordIdxSet = defaultdict(list)
		digitCorpus = self.digitCorpus

		try:
			wordIdxSet[''] = self.vocabIndex[word]
		except:
			pass

		for key in polySet:
			try:
				wordIdxSet[key] = self.vocabIndex[word + key]
			except:
				continue

		if len(wordIdxSet) == 0 or word in self.funcWords:
			# self.sem.release()
			return contextList

		count = 0

		for fdir in os.listdir(digitCorpus):

			for fname in os.listdir(digitCorpus+fdir):

				f = open(digitCorpus + fdir + '/' + fname, 'r')
				count += 1
				docLen = struct.unpack('i', f.read(4))[0]
				docIndex = array.array('i')
				docIndex.fromfile(f, docLen)
				docIndex = np.reshape(docIndex, (docLen))
				f.close()

				for key, wordIdx in wordIdxSet.iteritems():
					wordIndices = np.where(docIndex == wordIdx)
					for idx in wordIndices[0]:
						idxSet = range(idx-window, idx)
						idxSet.extend(range(idx+1, idx+1+window))
						try:
							contextList[key].append(docIndex[idxSet])
						except:
							pass

				del docIndex
				del wordIndices

		logging.info("%s done collecting corpus" % (word,))
		# self.sem.release()
		return contextList

	def pcaContexts(self, idxList, idx=-1, contextMatrix=None):
		'''
		input: context indices
		output: pca vectors
		'''

		vecs = self.vecMatrix[np.array(idxList)]
		# randIdx = np.random.randint(0, self.vocabSize, size=(1,), dtype='i')
		# vecs = self.vecMatrixNorm[randIdx]
		pca = PCA(n_components=self.pcaRank)
		pca.fit(vecs)
		contextVecs = pca.components_[0:self.pcaRank]

		if idx >= 0:
			contextMatrix[idx] = contextVecs

		del vecs

		return contextVecs, sum(pca.explained_variance_ratio_)

	def getCentVec(self, contextVecs):


		sample, rank, dim = contextVecs.shape
		contexts = np.reshape(contextVecs, (sample * rank, dim))
		pca = PCA(n_components=1)
		pca.fit(contexts)
		return pca.components_[0]

	def kMeansSubspace(self, contextList, figFile=None, ):

		'''
		input: list of context indices
		output: k-means centroids
		'''

		pcaRank = self.pcaRank
		contextSize = self.contextSize
		vecDim = self.vecDim
		adapt = self.adapt
		maxSenNum = self.maxSenNum
		kmeansIterMax = self.kmeansIterMax

		logging.info('Starting k-means size: %d' % len(contextList) )

		if len(contextList) >= contextSize:
			idxSet = list(np.random.choice(len(contextList), contextSize, replace=False))
			sampleContextList = [contextList[i] for i in idxSet]
			del contextList
			contextList = sampleContextList

		contextVecs = np.zeros((len(contextList), pcaRank, vecDim))
		threads = []
		start = time.clock()
		for i in xrange(len(contextList)):
			
			t = threading.Thread(target = self.pcaContexts, args = (contextList[i], i, contextVecs))
			t.start()
			threads.append(t)

		for t in threads:
			t.join()

		end = time.clock()

		logging.info("Analysing Context Time: %f" % (end - start))

		if adapt:
			recur = xrange(1, maxSenNum+1)
		else:
			recur = [maxSenNum]

		sample, rank, dim = contextVecs.shape
		fig, ax = plt.subplots(len(recur), 2, figsize=((7, 3*len(recur))))
		ax = ax.ravel()
		lastErr = 0

		for senIdx, senNum in enumerate(recur):

			# senNum = senIdx+1
			minErr = float('inf')

			for ranIdx in xrange(5):

				tempSenseVecs = np.random.normal(size=(senNum, dim)) 
				tempSenseVecs = normalizeMatrix(tempSenseVecs)
				postErr = 0
				curErr = float('inf')
				iterNum = 0
				while True:
					# if debug:
					# 	print >>sys.stdout, senNum, 'kmeans correlation:', np.dot(senseVecs, senseVecs.T)
					if iterNum > kmeansIterMax:
						break
					iterNum += 1
					postErr = curErr
					# cluster subspaces
					errSum = 0

					coeffs = np.dot(contextVecs, tempSenseVecs.T)
					d = np.sum(coeffs ** 2, 1)
					errs = 1 - np.max(d, axis=1)
					## remove numerical negative term
					errs = (errs + np.absolute(errs))/2
					errSum = sum(errs)
					labels = np.argmax(d, axis=1)

					# update centers of subspaces
					# logging.info("%f\t%d" % (errSum/sample, iterNum))
					curErr = errSum

					if abs(postErr - curErr) < 1e-5 * sample:
						break

					for s in xrange(senNum):
						clusterIdx = np.where(labels == s)
						if len(labels[clusterIdx]) == 0:
							continue
						tempSenseVecs[s] = self.getCentVec(contextVecs[clusterIdx])

				if curErr <= minErr:
					minErr = curErr
					senseVecs = tempSenseVecs

			logging.info('K-means error %0.4f for %d clusters.' % (minErr/sample, senNum))

			commonParams = dict(bins=40, 
			                     range=(0, 1), 
			                     normed=True,
			                     histtype='step')


			if minErr >= lastErr * 0.8 and lastErr != 0:
				break

			lastErr = minErr
			lastSenseVecs = senseVecs

			coeffs = np.dot(contextVecs, senseVecs.T)
			d = np.sum(coeffs ** 2, 1)
			errs = 1 - np.max(d, axis=1)
			## remove numerical negative term
			errs = (errs + np.absolute(errs))/2
			errSum = sum(errs)
			errors = [errs ** 0.5]
			for i in xrange(len(senseVecs)):
				senVec = senseVecs[i].reshape((1, vecDim))
				coeffs = np.dot(contextVecs, senVec.T)
				d = np.sum(coeffs ** 2, 1)
				errs = 1 - np.max(d, axis=1)
				errs = (errs + np.absolute(errs))/2
				errors.append((errs) ** 0.5)

			ax[2*senIdx].hist(errors, **commonParams)
			ax[2*senIdx].set_title('l_2 error = %f' % (errSum/sample,))

			senseCor = np.abs(np.dot(senseVecs, senseVecs.T))
			im = ax[2*senIdx + 1].imshow(senseCor, interpolation='nearest', vmin=0, vmax=1)
			ax[2*senIdx + 1].xaxis.set_visible(False)
			ax[2*senIdx + 1].yaxis.set_visible(False)
			# ax[2*senIdx + 1].set_title(' '.join([str(len(labels[np.where(labels == s)])) for s in xrange(senNum)]))


		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)

		if figFile:
			plt.savefig(figFile, format='pdf', transparent=True)
		plt.close()

		return lastSenseVecs

	def minClusterIdx(self, arrayList, subspace):

		coeffs = np.dot(subspace, arrayList.T)
		d = (np.sum(coeffs ** 2, 0))
		return 1-np.max(d), np.argmax(d)

	def clusterDist(self, arrayList, subspace):
		coeffs = np.dot(subspace, arrayList.T)
		d = (np.sum(coeffs ** 2, 0))
		return (1 - d) ** 0.5

	def findContexts(self, polysemy):

		start = time.clock()
		figFile = self.algoDir + 'figures/'+polysemy+'.pdf'
		vecFile = self.algoDir + 'vecs/'+polysemy+'.bin'

		if isfile(vecFile):
			logging.info('%s exists.' % (polysemy,))
			self.sem.release()
			return

		logging.info( "starting %s" % (polysemy))

		if isfile(self.preDir + 'vecs/' + polysemy + '.bin'):
			_, prePolySet, _ = cPickle.load(open(self.preDir + 'vecs/' + polysemy + '.bin', 'r'))
		else:
			prePolySet = list()

		contextListSet = self.findWord(polysemy, prePolySet)
		polySet = list()
		senseVecs = list()
		end = time.clock()
		logging.info("%s" % ("Extracting Context Time: {}".format(end-start)))

		if len(contextListSet) == 0:
			self.sem.release()
			return 

		count = 0
		for key, contextList in contextListSet.iteritems():
			if len(contextList) == 0:
				continue
			tempSenseVecs = self.kMeansSubspace(contextList, figFile)
			for i in xrange(len(tempSenseVecs)):
				senseVecs.append(tempSenseVecs[i])
			count += 1
			for i in xrange(self.maxSenNum):
				polySet.append(key+'.'+str(i+1))

		for key in contextListSet.keys():
			polySet.append(key+'.0')

		senseVecs = np.array(senseVecs)

		fVecOut = open(vecFile, 'wb')
		cPickle.dump([polysemy, polySet, senseVecs], fVecOut)
		fVecOut.close()
		logging.info('Saved %s.' % polysemy)

		end = time.clock()
		logging.info( "%s" % ("Total Time: {}".format(end-start)))

		del contextListSet
		self.sem.release()
		return senseVecs

	#####################################################
	# disambiguation

	def disambiguate(self, centVecs, contextStr):

		context = contextStr.lower().split()
		try:
			start = context.index('<b>')
			end = context.index('</b>')
		except:
			return -1, -1, 0, ''
		contextId = list()
		retStr = ' '.join(context[start:end+1])
		idx = start
		count = 0
		while idx > 0:
			idx -= 1
			if context[idx] in self.funcWords or count >= self.window:
				retStr = '%s %s' % (context[idx], retStr)
				continue
			try:
				contextId.append(self.vocabIndex[context[idx]])
				retStr = '[%s] %s' % (context[idx], retStr)
				count += 1
			except:
				retStr = '%s %s' % (context[idx], retStr)
				pass
		idx = end
		count = 0
		while idx < len(context) - 1:
			idx += 1
			if context[idx] in self.funcWords or count >= self.window:
				retStr = '%s %s' % (retStr, context[idx])
				continue
			try:
				contextId.append(self.vocabIndex[context[idx]])
				count += 1
				retStr = '%s [%s]' % (retStr, context[idx])
			except:
				retStr = '%s %s' % (retStr, context[idx])
				pass
		if len(contextId) < self.pcaRank:
			return -1, -1, 0, retStr
		contextVecs, ratio = self.pcaContexts(contextId)
		err, label = self.minClusterIdx(centVecs, contextVecs)
		err = ((err + np.abs(err))/2) ** 0.5
		return err, label, ratio, retStr

	def disambiguate_soft(self, centVecs, contextStr):

		context = contextStr.lower().split()
		try:
			start = context.index('<b>')
			end = context.index('</b>')
		except:
			return -1, -1, 0, ''
		contextId = list()
		retStr = ' '.join(context[start:end+1])
		idx = start
		count = 0
		while idx > 0:
			idx -= 1
			if context[idx] in self.funcWords or count >= self.window:
				retStr = '%s %s' % (context[idx], retStr)
				continue
			try:
				contextId.append(self.vocabIndex[context[idx]])
				retStr = '[%s] %s' % (context[idx], retStr)
				count += 1
			except:
				retStr = '%s %s' % (context[idx], retStr)
				pass
		idx = end
		count = 0
		while idx < len(context) - 1:
			idx += 1
			if context[idx] in self.funcWords or count >= self.window:
				retStr = '%s %s' % (retStr, context[idx])
				continue
			try:
				contextId.append(self.vocabIndex[context[idx]])
				count += 1
				retStr = '%s [%s]' % (retStr, context[idx])
			except:
				retStr = '%s %s' % (retStr, context[idx])
				pass
		if len(contextId) < self.pcaRank:
			return -1, -1, 0, retStr
		contextVecs, ratio = self.pcaContexts(contextId)
		err = self.clusterDist(centVecs, contextVecs)
		prob = np.exp(-10 * err) # 1/(err1) #  
		prob /= np.sum(prob)
		return prob, ratio, retStr

	#####################################
	# labeling corpus

	def processCorpus(self, polyList, outDirectory, corpusDir, corpusName, decoding):

		wordList = list()
		senseVecs = list()
		labels = list()

		print >>sys.stdout, 'Start labeling corpus.'

		for poly in polyList:
			if os.path.isfile(self.algoDir + 'vecs/' + poly + '.bin'):
				f = open(self.algoDir + 'vecs/' + poly + '.bin', 'rb')
			else:
				continue
			polysemy, polySet, polyVecs = cPickle.load(f)
			f.close()
			wordList.append(polysemy)
			senseVecs.append(polyVecs)
			labels.append(polySet)


		print >>sys.stdout, 'Done loading sense vectors.'


		count = 0
		processes = []
		inCorpus = self.corpusDir
		outCorpus = outDirectory + corpusDir
		# os.system('rm -rf '+curCorpus)

		for fdir in os.listdir(inCorpus):

			os.system('mkdir -p ' + outCorpus + fdir + '/')

			print >>sys.stdout, 'Labeling corpus from', fdir
			if fdir.startswith('.'):
				continue

			for fname in os.listdir(inCorpus + fdir):

				if os.path.isfile(outCorpus + fdir + '/' + fname):
					if os.path.getsize(outCorpus + fdir + '/' + fname) % io.DEFAULT_BUFFER_SIZE != 0:
						print >>sys.stdout, '%s/%s exists.' % (fdir, fname)
						continue
					if abs(time.time() - os.path.getmtime(outCorpus + fdir + '/' + fname)) < 300:
						print >>sys.stdout, '%s/%s is now working' % (fdir, fname)
						continue
				else:
					print >>sys.stdout, '%s/%s labeling.' % (fdir, fname)

				self.sem.acquire()

				fin = open(inCorpus + fdir + '/' + fname, 'r')
				fout = open(outCorpus + fdir + '/' + fname , 'w')
				# print >>sys.stdout, fdir, fname
				t = multiprocessing.Process(target = self.disambiguateCorpus_thread, args=(fin, fout, wordList, senseVecs, labels, decoding))
				processes.append(t)
				t.start()
				

				# if self.debug:
				# 	break
					
			# if self.debug:
			# 	break

		for t in processes:
			t.join()

		print >>sys.stdout, 'Done post-processing corpus.'
				
	def disambiguateCorpus_thread(self, fpreCorpus, fpostCorpus, wordList, senseVecs, labels, decoding):

		start = time.clock()
		doc = fpreCorpus.read().rstrip().split()
		fpreCorpus.close()

		# print doc

		for i, word in enumerate(doc):
			try: 
				wordIdx = wordList.index(word.split('.')[0])
			except:
				# if debug:
				# 	print 'not found', word
				print >>fpostCorpus, word,
				continue

			k = i 
			count = 0
			contextIdx = list()
			while count < self.window and k > 0:
				k -= 1
				if doc[k] in self.funcWords:
					continue
				try:
					contextIdx.append(self.vocabIndex[doc[k]])
					count += 1
				except:
					pass

			k = i
			count = 0
			while count < self.window and k < len(doc)-2:
				k += 1
				if doc[k] in self.funcWords:
					continue
				try:
					contextIdx.append(self.vocabIndex[doc[k]])
					count += 1
				except:
					pass

			contextVecs, _ = self.pcaContexts(contextIdx)
			
			if decoding == 'hard':
				err, label = self.minClusterIdx(senseVecs[wordIdx], contextVecs)
				err = ((err + np.abs(err))/2) ** 0.5
				if err <= 0.6:
					# if debug:
					# 	print 'label', word.split('.')[0]  + str(labels[wordIdx][label])
					print >>fpostCorpus, word.split('.')[0] + str(labels[wordIdx][label]),
				else:
					# if debug:
					# 	print 'notlabel', word + '.0' 
					print >>fpostCorpus, word + '.0',
			elif decoding == 'soft':
				err = self.clusterDist(senseVecs[wordIdx], contextVecs)
				prob = np.exp(-10 * err) # 1/(err1) #  
				prob /= np.sum(prob)
				label = np.random.choice(labels[wordIdx][:-1], p = prob)
				print >>fpostCorpus, word.split('.')[0] + label,

			
		fpostCorpus.close()
		end = time.clock()
		logging.info('Time: %f' % (end-start))

		self.sem.release()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', default=1, type=int)
	parser.add_argument('--normalize', default=0, type=int)
	parser.add_argument('--dirBase', default='/projects/csl/viswanath/data/jiaqimu2/wordnet/data/non-functional/post/', type=str)
	parser.add_argument('--maxSenNum', default=5, type=int)
	parser.add_argument('--maxIter', default=1, type=int)
	parser.add_argument('--refresh', default=0, type=int)
	parser.add_argument('--iterNum', default=1, type=int)
	parser.add_argument('--adapt', default=0, type=int)
	parser.add_argument('--primary', default=0, type=int)
	args = parser.parse_args()

	funWordsFile = '/projects/csl/viswanath/data/jiaqimu2/wordnet/data/funcWords.txt'

	# vocab = Vocab(vecDim, directory, corpusDir, corpusName, vocabInputFile, vecInputFile, debug, funWordsFile)
	vocab = Vocab(300, args.dirBase, 'corpus/', 'wikiCorpus.txt', 'vocab.txt', 'vectors.bin', True, funWordsFile)
	polyList = vocab.vocabList[0:500]
	# vocab.computeSenseVecs(polyList, digitCorpusDir, algoDir, '', 
	# 				       pcaRank, window, contextSize, adapt, maxSenNum, kmeansIterMax)
	vocab.computeSenseVecs(polyList, 'digitCorpus/', args.dirBase + 'kGrassMean/senNum-%d/' % args.maxSenNum, '',
						   4, 10, 10000, False, args.maxSenNum, 100)