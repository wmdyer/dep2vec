import argparse, os, pickle, re, subprocess, sys, tqdm
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models.word2vec import *

GRAPH = True
DEPREL = False

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

def pickle_vecs(model, ndim):
    vectors = {}
    print("converting to dict")
    for key in tqdm.tqdm(model.wv.vocab.keys()):
        vec = list(model.wv[key])
        vectors[key] = vec

    pkl_file = 'vectors.pkl'
    print("writing pickle data to " + pkl_file)
    pf = open(pkl_file, 'wb')
    pickle.dump(vectors, pf)
    pickle.dump(ndim, pf)
    pf.close()

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def make_sent(words, heads):
    sent = []
    for i in words.keys():
        sent.append(words[i])

    sents = []
    for h in heads.keys():
        for d in heads[h]:
            try:
                _sent = sent.copy()                            
                _sent[h-1] = words[d] + "_" + words[h]
                sents.append(_sent)
            except:
                pass
    return sents

def extract_sents(filename):

    sents = []

    print("extracting from " + filename)
    prev_idx = 0
    words = {}
    edges = {}
    #deprels = {}
    heads = defaultdict(list)

    f = open(filename, 'r')
    for line in tqdm.tqdm(f, total=line_count(filename)):
        cols = line.split('\t')
        
        if len(cols) > 1 and '-' not in cols[0] and '.' not in cols[0]:
            idx = int(cols[0])

            if idx < prev_idx:
                if GRAPH:
                    G = nx.DiGraph()
                    G.add_edges_from(([(edges[k], k) for k in edges.keys()]))
                    sents.append(graph2sents(G, words))
                else:
                    sents.append(make_sent(words, heads))

                words = {}
                heads = defaultdict(list)
                edges = {}
            
            words[idx] = cols[1].lower()
            edges[idx] = int(cols[6])
            hdx = int(cols[6])
            if idx not in heads[hdx]:
                heads[hdx].append(idx)
            prev_idx = idx

    if GRAPH:
        sents.append(graph2sents(G, words))
        return [item for sublist in sents for item in sublist]
    else:
        sents.append(make_sent(words, heads))
        return sents

def graph2sents(G, words):
    sents = []
    leaves = [x for x in G.nodes() if G.out_degree(x)==0]
    for leaf in leaves:
        path = []
        try:
            path = nx.shortest_path(G, 0, leaf)
        except:
            break

        sent = [words[int(x)] for x in path[::-1][:-1]]
        if len(sent) > 1:
            sents.append(sent)
                            
            for i in range(0,len(sent)-1):
                a = []
                try:
                    a.append(sent[0:i][0])
                except:
                    pass
                try:
                    a.append(sent[i]+"_"+sent[i+1])
                except:
                    pass
                try:
                    a.append(sent[i+2:][0])
                except:
                    pass
                if len(a) > 1:
                    sents.append(a)
    return sents

def main():

    parser = argparse.ArgumentParser(description='generate unigram and bigram vectors from a conllu file')
    parser.add_argument('-i', dest='input_file', type=str, required=True, nargs=1,
                        help='input CoNLL-U corpus (UTF-8)')
    parser.add_argument('-o', dest='output_file', type=str, required=False, nargs=1,
                        help='output file name')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs (default 10)')
    parser.add_argument('--dimension', type=int, default=100,
                        help='word2vec: dimensionality of feature vectors (default 100)')
    parser.add_argument('--window', type=int, default=5,
                        help='word2vec: maximum distance between current and predicted word (default 5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='word2vec: use this many worker threads to train the model (default 4)')
    parser.add_argument('--use-skipgram', action='store_true', default=True,
                        help='use skip-gram instead of the CBOW')
    
    args = parser.parse_args()
    sents = extract_sents(args.input_file[0])

    print('Beginning to build model...')

    if(args.use_skipgram):
        sgFlag = 1
    else:
        sgFlag = 0
        
    model = Word2Vec(size=args.dimension, min_count=0, window=args.window, workers=args.workers, sg=sgFlag)

    print('Building vocabulary...')
    model.build_vocab(sents)
    model.train(sents, total_examples=model.corpus_count, epochs=args.epochs)

    try:
        write_vecs(model, args.dimension, args.output_file[0])
    except:
        pickle_vecs(model, args.dimension)

if __name__ == '__main__':
    main()

