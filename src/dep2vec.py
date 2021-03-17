import argparse, math, pickle, re, sys, tqdm
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SISTER = '+'
PUNCT = True
DEPREL = False

def read_conllu(filename):
    print("Reading " + filename)
    df = pd.read_csv(filename, sep='\t', comment='#', header=None, dtype=str, quoting=3, engine='python', error_bad_lines=False, skip_blank_lines=True, index_col=None )
    df.columns = ['IDX', 'WORDFORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

    # remove MWEs
    df = df.loc[~df['IDX'].str.contains('-')]

    # remove PUNCT
    #df = df.loc[df['UPOS'] != 'PUNCT']

    # make IDX and HEAD numeric    
    df[['IDX', 'HEAD']] = df[['IDX', 'HEAD']].apply(pd.to_numeric, errors='coerce', downcast='float')

    df = df.reset_index(drop=True)
    
    # case-normalize wordform
    df['WORDFORM'] = df['WORDFORM'].str.lower()
    if DEPREL:
        df['WORDFORM'] = df['WORDFORM'] + "/" + df['DEPREL']

    return df

def write_vectors(df, outfilename, v, f, ndim):

    print("writing to " + outfilename)

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=100000)
    np.set_printoptions(sign=' ')
    np.set_printoptions(floatmode='fixed')

    for key in f:
        f[key] = np.reshape(f[key], (ndim,ndim))

    if SISTER == '*':
        f['ONES'] = np.ones((ndim,ndim))
        v['ONES'] = np.ones(ndim)
    else:
        f['ONES'] = np.zeros((ndim,ndim))
        v['ONES'] = np.zeros(ndim)
        
    f['IDENT'] = np.identity(ndim)
    v['IDENT'] = np.ones(ndim)
    
    outfile = open(outfilename, 'w')

    # get sentence boundaries
    sent_idx = df.loc[df['IDX'] == 1.0].index.values

    df['HEAD'] = df['HEAD'].astype(str)
    df['IDX'] = df['IDX'].astype(str)

    sentences = []
    outvecs = []
    trees = []

    scaler = MinMaxScaler(feature_range=(-1,1))

    # iterate through sentences
    for start in tqdm.tqdm(sent_idx):
        start_idx = np.where(sent_idx == start)[0][0]
        try:
            end = sent_idx[start_idx+1]
        except:
            end = len(df)
        dfs = df.iloc[start:end]
        dfs = dfs.reset_index(drop=True)
        
        dfs['EDGE'] = list(zip(dfs.HEAD, dfs.IDX))
        G = nx.DiGraph()
        G.add_edges_from(dfs['EDGE'].values)

        if True:
            dfs.set_index('IDX', inplace=True)
            words = dfs['WORDFORM'].to_dict()            
            _trees = []
            leaves = [x for x in G.nodes() if G.out_degree(x)==0]
            for leaf in leaves:
                try:
                    path = nx.shortest_path(G, '0', leaf)
                except:
                    try:
                        path = nx.shortest_path(G, '0.0', leaf)
                    except:
                        pass
                psent = []
                length = len(path[::-1][:-1]) - 1
                for i,x in enumerate(path[::-1][:-1]):
                    try:
                        if "'" not in words[x] and '"' not in words[x] and ((i < length and words[x] in f) or (i == length and words[x] in v)):
                            w = words[x]
                        else:
                            w = 'IDENT'
                        psent.append("f['" + w + "']")
                    except:
                        pass
                try:
                    psent[-1] = re.sub('f\[', 'v[', psent[-1])
                    _trees.append('(' + '@'.join(psent) + ')')
                except:
                    pass

            tree = "np.sum([" + ','.join(_trees) + "], axis=0)"
                
        else:
            tree = re.sub("@$", "", print_node(G, '0', "", dfs, v, f))

        try:
            vec = eval(tree)
            trees.append(tree)
            outvecs.append(vec)
            
            sentence = ' '.join(dfs['WORDFORM'].values.astype(str))
            sentences.append(sentence)
        except Exception as e:
            print(e)
            print(tree)
            exit()
            pass

    for i,sentence in enumerate(sentences):
        try:
            outvec = np.array2string(outvecs[i])[1:-1].replace('  ', ' ').strip()
            tree = trees[i]
            outfile.write('\t'.join([sentence, tree, outvec]) + "\n")
        except Exception as e:
            print(e)
    outfile.close()

def print_node(G, n, out, df, v, f):
    for i,s in enumerate(G.successors(n)):
        if i == 0:
            out += "("
        else:
            out += SISTER
        out = print_node(G, s, out, df, v, f)

        if i == len(list(G.successors(n))) - 1:
            out += ")@"

    if n != "0.0":
        w = df.loc[df['IDX'] == n]['WORDFORM'].values[0]
        h = df.loc[df['IDX'] == n]['HEAD'].values[0]
        if not PUNCT and df.loc[df['IDX'] == n]['UPOS'].values[0] == 'PUNCT':
            w = 'OOV'

        if h != '0.0':
            if w in f:
                out += "f['" + df.loc[df['IDX'] == n]['WORDFORM'].values[0] + "']"
            else:
                prev = SISTER
                i = -1
                try:
                    while out[i] not in ['@', SISTER]:
                        i += -1

                    prev = out[i]
                except:
                    pass
                if prev == '@':
                    out += "f['IDENT']"
                elif prev == SISTER:
                    out += "f['ONES']"
                else:
                    print(out)
                    print(i, prev)
                    exit()
        else:
            if w in v:
                out += "v['" + df.loc[df['IDX'] == n]['WORDFORM'].values[0] + "']"
            else:
                out += "v['ONES']"
                

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vectorize sentences in conllu format')
    parser.add_argument('-i', '--infile', dest='infile', nargs=1, required=True, help='conllu file')
    parser.add_argument('-o', '--outfile', dest='outfile', nargs=1, required=True, help='out file')
    args = parser.parse_args()
    
    conllu = read_conllu(args.infile[0])
    pkl_file = 'vectors.pkl'
    try:
        pf = open(pkl_file, 'rb')
        print("loading pickle data from " + pkl_file)
        v = pickle.load(pf)
        f = pickle.load(pf)
        ndim = pickle.load(pf)
        pf.close()
    except:
        print("ERROR: vectors.pkl doesn't exist")
        exit()
    
    write_vectors(conllu, args.outfile[0], v, f, int(math.sqrt(ndim)))
    
