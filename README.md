
We provide an implementation of $K$-Grassmeans for word sense induction, disambiguation, and representation. Details please refer to our paper:
- Jiaqi Mu, Suma Bhat, and Pramod Viswanath. "Geometry of Polysemy." arXiv preprint arXiv:1610.07569 (2016).

In src/, we provide three scripts:
  - induction.py: to get intersections for word sense induction directly from corpus.
  - induction-from-file.py: to get intersections for word sense induction from a given set of sentences.
  - representation.py: to generate a labeled corpus.

We provide three demos to try:
  - demo-induction.sh
  - demo-induction-from-file.sh
  - demo-representation.sh
  
When dealing with a training corpus, please chunk a large corpus into smaller files to avoid memory overflow. 

You will need to setup following parameters:
  
  - funcWordFile: a list of function words (an example is provided in data/)
  - polyListFile: a list of target polysemous words (an example is provided in data/)
  - directory: a base directory
  - vocabInputFile: an input vocabulary file (an example is provided in data/)
  - vecInputFile: a binary vector file (an example is provided in data/)
  - corpusPath: a directory to chunked corpus (an example is provided in data/)
  - algoPath: an output directory
