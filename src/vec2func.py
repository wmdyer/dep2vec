import numpy as np
import pandas as pd
import argparse, pickle, sys, tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

VEC_LIMIT = 100

def read_vectors(filename):
    print("\nLoading vectors from %s" % filename, file=sys.stderr)
    d = {}
    with open(filename) as infile:
        n, ndim = next(infile).strip().split()
        n = int(n)
        ndim = int(ndim)
        lines = list(infile)
        for line in tqdm.tqdm(lines):
            parts = line.strip().split(" ")
            numbers = list(map(float, parts[-ndim:]))
            #vec = torch.Tensor(numbers)
            wordparts = parts[:-ndim]
            word = " ".join(wordparts)
            d[word] = numbers

    return d, ndim

def read_conllu(filename):
    df = pd.read_csv(filename, sep="\t",error_bad_lines=False, engine='python', header=None, comment= '#', quoting=3)
    df.columns = ['IDX', 'WORDFORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'OTHER', 'MISC']
    return df

def calc_vecs(vectors):
    mvecs = {}

    deps = defaultdict(list)
    unigrams = {}

    print("assembling dependency relations")
    for key in tqdm.tqdm(vectors.keys()):
        if '_' in key:
            d = key.split('_')[0]
            h = key.split('_')[1]            
            if h not in deps[d]:
                deps[d].append(h)
        else:
            unigrams[key] = vectors[key]
                

                
    print("calculating functions")
    for d in tqdm.tqdm(deps.keys()):
        
        X = []
        y = []

        for h in deps[d]:
            try:
                X.append(vectors[h])
                y.append(vectors[d])
            except:
                pass

        if len(X) > 1:
            X = np.array(X)
            y = np.array(y)
            try:
                w = np.linalg.inv(X.T@X)@X.T@y
                mvecs[d] = w.flatten()
            except:
                pass
            
    return mvecs, unigrams

def write_vecs(d, ndim, outfilename):
    print("Saving vectors to " + outfilename)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000000)
    np.set_printoptions(sign=' ')
    np.set_printoptions(floatmode='fixed')
    np.set_printoptions(threshold=10000000)
    
    outfile = open(outfilename, 'w')
    outfile.write("%s %s\n"%(len(d.keys()), ndim**2))
    for key in tqdm.tqdm(d.keys()):
        vec = np.array2string(d[key])[1:-1].strip().replace('  ', ' ')
        outfile.write("%s %s\n"%(key, vec))
    outfile.close()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='make mods')
    parser.add_argument('-i', '--infile', dest='infile', nargs=1, required=True, help='input vector file')
    #parser.add_argument('-o', '--outfile', dest='outfile', nargs=1, required=True, help='output vector file')
    parser.add_argument('-c', '--conllu', dest='conllu', nargs=1, required=False, help='conllu file')
    args = parser.parse_args()

    vectors, ndim = read_vectors(args.infile[0])

    try:
        ud = read_conllu(args.conllu[0])
        d = calc_vecs_conllu(vectors, ud, args.conllu[0])
    except:
        d, vectors = calc_vecs(vectors)

    pkl_file = 'vectors.pkl'
    print("writing pickle data to " + pkl_file)
    pf = open(pkl_file, 'wb')
    pickle.dump(vectors, pf)
    pickle.dump(d, pf)
    pickle.dump(ndim**2, pf)
    pf.close()
    
    #write_vecs(d, ndim, args.outfile[0])

