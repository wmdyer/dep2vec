import argparse, os, subprocess, sys, tqdm
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models.word2vec import *

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

def read_conllu(filename):
    print("Reading " + filename)
    df = pd.read_csv(filename, sep='\t', comment='#', header=None, dtype=str, quoting=3, engine='python', error_bad_lines=False, skip_blank_lines=True, index_col=None )
    df.columns = ['IDX', 'WORDFORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    
    # remove MWEs
    df = df.loc[(~df['IDX'].str.contains('-')) & (~df['IDX'].str.contains('\.'))]
                    
    # make IDX and HEAD numeric
    df[['IDX', 'HEAD']] = df[['IDX', 'HEAD']].apply(pd.to_numeric, errors='coerce', downcast='integer')

    df = df.dropna().reset_index(drop=True)

    # case-normalize both lemma and wordform
    df['LEMMA'] = df['LEMMA'].str.lower()
    df['WORDFORM'] = df['WORDFORM'].str.lower()
    
    return df

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def extract_sents(filename):

    sents = []

    print("extracting from " + filename)
    prev_idx = 0
    words = {}
    edges = {}
    heads = defaultdict(list)

    f = open(filename, 'r')
    for line in tqdm.tqdm(f, total=line_count(filename)):
        cols = line.split('\t')
        
        if len(cols) > 1 and '-' not in cols[0] and '.' not in cols[0]:
            idx = int(cols[0])

            if idx < prev_idx:
                G = nx.DiGraph()
                G.add_edges_from(([(edges[k], k) for k in edges.keys()]))
        
                sents.append(graph2sents(G, words))
                words = {}
                edges = {}

            else:
                words[idx] = cols[1].lower()
                edges[idx] = int(cols[6])
                
            prev_idx = idx

    return [item for sublist in sents for item in sublist]

    # get sentence boundaries
    sent_idx = df.loc[df['IDX'] == 1].index.values

    df['HEAD'] = df['HEAD'].astype(int)
    df['IDX'] = df['IDX'].astype(int)

    G = nx.DiGraph()

    # iterate through sentences
    for start in tqdm.tqdm(sent_idx):
        start_idx = np.where(sent_idx == start)[0][0]
        try:
            end = sent_idx[start_idx+1]
        except:
            end = len(df)
        dfs = df.iloc[start:end]
        dfs = dfs.reset_index(drop=True)

        dfs['NODE'] = dfs['IDX']

        dfs['EDGE'] = list(zip(dfs.HEAD, dfs.NODE))

        G = nx.DiGraph()
        G.add_edges_from(dfs['EDGE'].values)
        dfs.set_index('IDX', inplace=True)
        words = dfs['WORDFORM'].to_dict()
        
        sents.append(graph2sents(G, words))

    return [item for sublist in sents for item in sublist]

def graph2sents(G, words):
    sents = []
    leaves = [x for x in G.nodes() if G.out_degree(x)==0]
    for leaf in leaves:
        try:
            path = nx.shortest_path(G, 0, leaf)
            sent = [words[int(x)-1] for x in path[::-1][:-1]]
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
        except:
            pass

    return sents

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
    parser.add_argument('--dimension', type=int, default=100,
                        help='word2vec: dimensionality of feature vectors (default 100). doc2vec mode may demand a higher value.')
    parser.add_argument('--window', type=int, default=5,
                        help='word2vec: maximum distance between current and predicted word (default 5). doc2vec mode may demand a higher value.')
    parser.add_argument('--workers', type=int, default=4,
                        help='word2vec: use this many worker threads to train the model (default 4)')
    parser.add_argument('--use-skipgram', action='store_true', default=True,
                        help='Use skip-gram instead of the default CBOW.')
    
    args = parser.parse_args()

    #df = read_conllu(args.input_file)
    sents = extract_sents(args.input_file)


    #trainLabeledSentences = []
    
    #trainingCorpus = ConllFile(keepMalformed=True,
    #                           checkParserConformity=False,
    #                           projectivize=False,
    #                           enableLemmaMorphemes=False,
    #                           compatibleJamo=True)
    
    #fd = open(args.input_file, 'r', encoding='utf-8')
    #trainingCorpus.read(fd.read())
    #fd.close()

    #for sent in trainingCorpus.sentences:
    #    unigrams = []
    #    bigrams = []
    #    for token in sent.tokens:
    #        dep = token.FORM.lower()
    #        head = sent.tokens[token.HEAD].FORM.lower()
    #        unigrams.append(token.FORM.lower().strip())
    #        bigrams.append('_'.join([dep, head]).strip())
    #    trainLabeledSentences.append(unigrams)
        #trainLabeledSentences.append(bigrams)

    #print(trainLabeledSentences)
    #exit()
            
    print('Beginning to build model...')

    if(args.use_skipgram):
        sgFlag = 1
    else:
        sgFlag = 0
        
    model = Word2Vec(size=args.dimension, min_count=0, window=args.window, workers=args.workers, sg=sgFlag)

    print('Building vocabulary...')
    model.build_vocab(sents)
    model.train(sents, total_examples=model.corpus_count, epochs=args.epochs)
    
    write_vecs(model, args.dimension, args.output_file)

if __name__ == '__main__':
    main()

