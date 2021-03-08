# dep2vec

a method for the compositional vectorization of a multi-word string

## background
While BERT ([Devlin et al., 2019](https://www.aclweb.org/anthology/N19-1423/)) and other flavors of transformers seem to be the current SOTA for vectorized representations of sentences or other multi-word strings, they ignore the compositionality of language (Frege, 1892). That is, rather than treating a sentence linearly and generating a vector representation based on the surface order of words, we can instead vectorize the dependency structure represented by the surface order ([Mitchell & Lapata, 2008](https://www.aclweb.org/anthology/P08-1028/)). 

We begin with the idea that nouns are vectors and adjectives are functions (linear maps) operating on vectors ([Baroni & Zamparelli, 2010](https://www.aclweb.org/anthology/D10-1115/)). In this framework, an adjective serves to map an *n*-dimensional **N** vector to an *n*-dimensional **AN** vector; as such, the adjective function *a* is represented by an *n*x*n* tensor, which is set of weights resulting from a least-squares training process on multiple **N** → **AN** instances learned from a corpus ([Guevara, 2010](https://www.aclweb.org/anthology/W10-2805)). Thus the matrix-multiplication between *a* and **N** yields **AN**; or more succinctly: *a*@**N** → **AN**.

This paper proposes two extensions of the *a*(**N**) → **AN** idea: (1) expand the notion to all heads and dependents regardless of syntactic category such that roots are vectors and dependents are functions; and (2) perform Hadamard or element-wise multiplication between sister dependents.

For example, the below sentence and its associated dependency parse (from [Universal Depedencies](https://universaldependencies.org/introduction.html))

![Image of dependency graph](https://github.com/wmdyer/dep2vec/blob/main/img/ud.png)

would be vectorized as ((*the*@*dog*)\*(*was*)\*((*by*\**the*)@*cat*)\**.*)@**chased**, where **chased** is a vector and all other words are trained functions.

## code usage

1. ud2vec.py: generate vectors for unigrams and bigrams from a conllu file
```
usage: ud2vec.py [-h] [--epochs EPOCHS]
                 [--min-sentence-length MIN_SENTENCE_LENGTH]
                 [--dimension DIMENSION] [--window WINDOW] [--workers WORKERS]
                 [--min-word-occurrence MIN_WORD_OCCURRENCE] [--use-skipgram]
                 [--min-word-length MIN_WORD_LENGTH]
                 input_file output_file
```

2. vec2func.py: learn dependent functions
```
usage: vec2func.py [-h] -i INFILE -o OUTFILE
```

3. dep2vec.py: vectorize string in conllu format
```
usage: dep2vec.py [-h] -i INFILE -o OUTFILE -v VECTORS -f FUNCTIONS
```
