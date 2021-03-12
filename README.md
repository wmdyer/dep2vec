# dep2vec

a method for the compositional vectorization of a multi-word string

## background
While BERT ([Devlin et al., 2019](https://www.aclweb.org/anthology/N19-1423/)) and other flavors of contextual embeddings seem to be the current SOTA for vectorized representations of sentences or other multi-word strings, they ignore the compositionality of language (Frege, 1892). That is, rather than treating a sentence linearly and generating a vector representation based on the surface order of words, we can instead vectorize the dependency structure represented by the surface order ([Mitchell & Lapata, 2008](https://www.aclweb.org/anthology/P08-1028/)). 

We begin with the idea that nouns are vectors and adjectives are functions (linear maps) operating on vectors ([Baroni & Zamparelli, 2010](https://www.aclweb.org/anthology/D10-1115/)). In this framework, an adjective serves to map an *n*-dimensional **N** vector to an *n*-dimensional **AN** vector; as such, the adjective function *a* is represented by an *n*x*n* tensor, which is set of weights resulting from a least-squares training process on multiple **N** → **AN** instances learned from a corpus ([Guevara, 2010](https://www.aclweb.org/anthology/W10-2805)). Thus the matrix-multiplication between *a* and **N** yields **AN**; or more succinctly: *a*@**N** → **AN**.

How to properly determine the vector representation for an adjective-noun pair, **AN**, is unclear; more generally, the embedding of a bigram, whether linear or within a dependency pair, is an open question. [Mitchell & Lapata (2008)](https://www.aclweb.org/anthology/P08-1028/) add or multiply **A** and **N** to yield **AN**, while [Erk & Padó (2008)](https://www.aclweb.org/anthology/D08-1094/) use the centroid of verbs that a noun tends to be dependent on to compose **VN**. [Baroni & Zamparelli (2010)](https://www.aclweb.org/anthology/D10-1115/) and [Guevara (2010)](https://www.aclweb.org/anthology/W10-2805) suggest calculating the embedding of co-occurring adjectives and nouns in a corpus directly, as though they were unigrams.

## proposal
This project proposes three extensions of the *a*(**N**) → **AN** idea: (1) expand the notion to all heads and dependents regardless of syntactic category such that roots are vectors and dependents are functions; (2) calculate unigram and bigram vectors wholly within a dependency framework; and (3) sum the associated matrices (functions) of sister dependents.

For example, we have the below sentence and its associated dependency parse (from [Universal Depedencies](https://universaldependencies.org/introduction.html)).

![Image of dependency graph](https://github.com/wmdyer/dep2vec/blob/main/img/ud.png)

A string can be parsed by something like [UDPipe](https://github.com/ufal/udpipe) to reveal its dependency structure. The context for dependent words is simply the list of paths from each of the leaf nodes to the root, unigrams being the words themselves and bigrams each pair of adjacent words. Thus the contextual frames for the sentence above are

- the dog chased
- the_dog chased
- the dog_chased
- was chased
- by cat chased
- by_cat chased
- by cat_chased
- the cat chased
- the_cat chased
- the cat_chased
- . chased

Finally, the vectorized representation of the sentence is the result of evaluating ((*the*@*dog*)+(*was*)+((*by*+*the*)@*cat*)+*.*)@**chased**, where **chased** is a vector and all other words are trained functions.

## code usage

1. genvecs.py: generate unigram and bigram vectors via word2vec where context is within dependency structure
```
usage: genvecs.py [-h] [--epochs EPOCHS] [--dimension DIMENSION]
                  [--window WINDOW] [--workers WORKERS] [--use-skipgram]
                  input_file output_file
genvecs.py: error: the following arguments are required: input_file, output_file
```

2. vec2func.py: learn functions for dependent words
```
usage: vec2func.py [-h] -i INFILE -o OUTFILE
```

3. dep2vec.py: vectorize string in conllu format
```
usage: dep2vec.py [-h] -i INFILE -o OUTFILE -v VECTORS -f FUNCTIONS
```

## evaluation
TODO: measure performance of generated vectors on tasks such as paraphrase detection or machine translation and compare to transformer-generated vectors
