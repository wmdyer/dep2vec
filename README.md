# dep2vec

a method for the compositional vectorization of a multi-word string

### background
While BERT ([Devlin et al., 2019](https://www.aclweb.org/anthology/N19-1423/)) and other flavors of transformers seem to be the current SOTA for vectorized representations of sentences or other multi-word strings, they ignore the compositionality of language (Frege, 1892). That is, rather than treating a sentence linearly and generating a vector representation based on the surface order of words, we can instead vectorize the dependency structure represented by the surface order ([Mitchell & Lapata, 2008](https://www.aclweb.org/anthology/P08-1028/)). 

We begin with the idea that nouns are vectors and adjectives are functions (linear maps) operating on vectors ([Baroni & Zamparelli, 2010](https://www.aclweb.org/anthology/D10-1115/)). In this framework, an adjective serves to map an *n*-dimensional **N** vector to an *n*-dimensional **AN** vector; as such, the adjective function *a* is represented by an *n*x*n* tensor, which is set of weights resulting from a least-squares training process on multiple **N** → **AN** instances learned from a corpus ([Guevara, 2010](https://www.aclweb.org/anthology/W10-2805)). Thus the matrix-multiplication between *a* and **N** yields **AN**; or more succinctly: *a*\***N** → **AN**.

This paper proposes two extensions of the *a*(**N**) → **AN** idea: (1) expand the notion to all heads and dependents regardless of syntactic category such that roots are vectors and dependents are functions; and (2) perform Hadamard or element-wise multiplication between sister dependents.

For example, a sentence like 'the frog ate many small flies
