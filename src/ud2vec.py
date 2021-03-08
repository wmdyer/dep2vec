# take a CoNLL corpus and train word/doc embeddings
import argparse
import os
import sys
import tqdm
import numpy as np
from conll_utils import *
from gensim.models.word2vec import *

# random
from random import shuffle

# necessary for seeing logs
import logging

def write_vecs(model, ndim, outfilename):
    print("Saving vectors to " + outfilename)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=100000)
    np.set_printoptions(sign=' ')
    np.set_printoptions(floatmode='fixed')
    
    outfile = open(outfilename, 'w')
    outfile.write("%s %s\n"%(len(model.wv.vocab), ndim))
    for key in tqdm.tqdm(model.wv.vocab.keys()):
        vec = np.array2string(model.wv[key])[1:-1].strip().replace('  ', ' ')
        k = key.strip().replace(' ', '')
        outfile.write("%s %s\n"%(k, vec))
    outfile.close()

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process CoNLL-U corpus and generate a word2vec model.')
    # Required positional argument
    parser.add_argument('input_file', type=str,
                        help='Input CoNLL-U corpus (UTF-8)')
    parser.add_argument('output_file', type=str,
                        help='Base output filename of word2vec model (gensim)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs (default 10). Shuffle sentences and re-train during each training epoch.')
    parser.add_argument('--min-sentence-length', type=int, default=5,
                        help='If sentence is shorter than N Eojeols (full words), it will not be processed for inclusion in the word2vec model (default 5)')
    parser.add_argument('--dimension', type=int, default=100,
                        help='word2vec: dimensionality of feature vectors (default 100). doc2vec mode may demand a higher value.')
    parser.add_argument('--window', type=int, default=5,
                        help='word2vec: maximum distance between current and predicted word (default 5). doc2vec mode may demand a higher value.')
    parser.add_argument('--workers', type=int, default=4,
                        help='word2vec: use this many worker threads to train the model (default 4)')
    parser.add_argument('--min-word-occurrence', type=int, default=5,
                        help='word2vec: ignore all words with total frequency lower than this (default 5)')
    parser.add_argument('--use-skipgram', action='store_true', default=True,
                        help='Use skip-gram instead of the default CBOW.')
    parser.add_argument('--min-word-length', type=int, default=0,
                        help='word2vec: ignore all words with a length lower than this (default 0).')
    #parser.add_argument('--char2vec', action='store_true', default=False,
    #                    help='Create char2vec model (make all words their own chars).')
    
    trainLabeledSentences = []
    
    args = parser.parse_args()
    
    trainingCorpus = ConllFile(keepMalformed=True,
                               checkParserConformity=False,
                               projectivize=False,
                               enableLemmaMorphemes=False,
                               compatibleJamo=True)
    
    fd = open(args.input_file, 'r', encoding='utf-8')
    trainingCorpus.read(fd.read())
    fd.close()

    for sent in trainingCorpus.sentences:
        unigrams = []
        bigrams = []
        for token in sent.tokens:
            dep = token.FORM.lower()
            head = sent.tokens[token.HEAD].FORM.lower()
            unigrams.append(token.FORM.lower().strip())
            bigrams.append('_'.join([dep, head]).strip())
        trainLabeledSentences.append(unigrams)
        trainLabeledSentences.append(bigrams)        
            
    print('Beginning to build model...')

    if(args.use_skipgram):
        sgFlag = 1
    else:
        sgFlag = 0
        
    model = Word2Vec(size=args.dimension, min_count=args.min_word_occurrence, window=args.window, workers=args.workers, sg=sgFlag)

    print('Building vocabulary...')
    model.build_vocab(trainLabeledSentences)
    model.train(trainLabeledSentences, total_examples=model.corpus_count, epochs=args.epochs)
    
    write_vecs(model, args.dimension, args.output_file)

if __name__ == '__main__':
    main()

