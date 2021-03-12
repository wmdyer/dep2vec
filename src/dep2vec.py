import argparse, math, pickle, re, sys, tqdm
import networkx as nx
import numpy as np
import pandas as pd

SISTER = '+'

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
    
    # case-normalize both lemma and wordform
    df['LEMMA'] = df['LEMMA'].str.lower()
    df['WORDFORM'] = df['WORDFORM'].str.lower()

    return df

def read_vectors(filename):
    print("\nLoading data from %s" % filename, file=sys.stderr)
    d = {}
    with open(filename) as infile:
        n, ndim = next(infile).strip().split()
        n = int(n)
        ndim = int(ndim)
        lines = list(infile)
        for line in tqdm.tqdm(lines):
            parts = line.strip().split(" ")
            numbers = np.array(list(map(float, parts[-ndim:])))
            wordparts = parts[:-ndim]
            word = " ".join(wordparts)
            d[word] = numbers

    return d, ndim

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

        try:
            tree = re.sub("@$", "", print_node(G, '0.0', "", dfs, v, f))
            tree = '(2*v['.join(tree.rsplit('f[', 1)) + ')'

            if tree[0:2] == '(v':
                tree = tree + SISTER + "v['ONES']"

            vec = eval(tree)
                
            outvec = np.array2string(vec)[1:-1].replace('  ', ' ')
            
            sentence = ' '.join(dfs['WORDFORM'].values.astype(str))

            outfile.write('\t'.join([sentence, tree, outvec]) + "\n")
        except Exception as e:
            print(e)
            print(G.nodes)
            pass

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
        w = re.escape(df.loc[df['IDX'] == n]['WORDFORM'].values[0])

        if len(G.nodes) != len(out.split('[')):
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
                out += "(2*v['" + df.loc[df['IDX'] == n]['WORDFORM'].values[0] + "'])"
            else:
                out += "v['ONES']"
                

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make mods')
    parser.add_argument('-i', '--infile', dest='infile', nargs=1, required=True, help='conllu file')
    parser.add_argument('-o', '--outfile', dest='outfile', nargs=1, required=True, help='out file')
    parser.add_argument('-v', '--vectors', dest='vectors', nargs=1, required=True, help='vectors file')
    parser.add_argument('-f', '--functions', dest='functions', nargs=1, required=True, help='functions file')
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
        v, ndim = read_vectors(args.vectors[0])
        f, ndim = read_vectors(args.functions[0])

        print("writing pickle data to " + pkl_file)
        pf = open(pkl_file, 'wb')
        pickle.dump(v, pf)
        pickle.dump(f, pf)
        pickle.dump(ndim, pf)
        pf.close()
    
    write_vectors(conllu, args.outfile[0], v, f, int(math.sqrt(ndim)))
    
