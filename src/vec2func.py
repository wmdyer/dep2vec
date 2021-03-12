import numpy as np
import pandas as pd
import argparse, sys, tqdm
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
    print("Calculating vectors")
    
    mvecs = {}

    deps = []
    for key in vectors.keys():
        if '_' not in key:
            deps.append(key)
    
    for dep in tqdm.tqdm(deps):
        
        X = []
        y = []
        
        for k in vectors.keys():
            if k.startswith(dep+"_"):
                try:
                    X.append(vectors[k.split('_')[1]])
                    y.append(vectors[k])
                except:
                    pass
            if len(X) > VEC_LIMIT:
                break

        if len(X) > 1:
            X = np.array(X)
            y = np.array(y)
            try:
                w = np.linalg.inv(X.T@X)@X.T@y
                mvecs[dep] = w.flatten()
            except:
                pass
            
    return mvecs

def calc_vecs_conllu(vectors, ud, udname):
    mvecs = {}

    words = {}
    deps = defaultdict(list)
    relations = {}

    X = []
    y = []

    total = len(ud)
    print("Getting dependency relations from " + udname)
    for i,row in tqdm.tqdm(ud.iterrows(), total=total):
        if '-' not in row['IDX'] and '.' not in row['IDX']:
            idx = row['IDX']
            if int(row['IDX']) == 1 and len(relations) > 0:
                for key in relations.keys():
                    try:
                        head = words[relations[key]]
                        dep = words[key]
                        if head not in deps[dep]:
                            deps[dep].append(head)
                    except:
                        pass
                words = {}
                relations = {}
            else:
                try:
                    words[idx] = row['WORDFORM'].lower()
                    relations[idx] = row['HEAD']
                except:
                    pass

    print("Calculating functions")
    #scaler = MinMaxScaler(feature_range=(-1.5,1.5))
    for d in tqdm.tqdm(deps):
        
        X = []
        y = []

        for h in deps[d]:
            try:
                y.append(np.add(vectors[h],vectors[d])/2)
                X.append(vectors[h])
            except:
                pass

            if len(X) > VEC_LIMIT:
                break

        if len(X) > 1:
            X = np.array(X)
            y = np.array(y)
            try:
                w = np.linalg.inv(X.T@X)@X.T@y
                #w = w.reshape(-1,1)
                #scaler.fit(w)
                #mvecs[d] = scaler.transform(w).flatten()
                mvecs[d] = w.flatten()
            except Exception as e:
                print(e)
                exit()
                pass
            
    return mvecs


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
    parser.add_argument('-o', '--outfile', dest='outfile', nargs=1, required=True, help='output vector file')
    parser.add_argument('-c', '--conllu', dest='conllu', nargs=1, required=False, help='conllu file')
    args = parser.parse_args()

    vectors, ndim = read_vectors(args.infile[0])

    try:
        ud = read_conllu(args.conllu[0])
        d = calc_vecs_conllu(vectors, ud, args.conllu[0])
    except:
        d = calc_vecs(vectors)
    write_vecs(d, ndim, args.outfile[0])

