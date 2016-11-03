This dataset contains 50 polysemous words, each word is associated with a few contexts, and each context is associated with a label. 

The contexts for a target word are provided in train/, each context is associated with an index, the target word is marked with <b> </b>, e.g.:
- abstract.v.1	A functional (e.g. Scheme) programmer would create functions representing both elements and behaviors of the airport. A metalinguistic programmer would <b> abstract </b> the problem by creating a new language for modelling an airport with its own primitives and operations. The language 

The labels are provided in all.key. Each line contains a word, a context index, and a context label, e.g.:
- abstract abstract.94 abstract.2
