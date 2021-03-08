import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import pandas
import argparse, sys, tqdm

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

    print("Loaded.", file=sys.stderr)
    return d, ndim

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

        if len(X) > 1:
            X = np.array(X)
            y = np.array(y)
            try:
                w = np.linalg.inv(X.T@X)@X.T@y
                mvecs[dep] = w.flatten()
            except:
                pass
            
    return mvecs
                    
def write_vecs(d, ndim, outfilename):
    print("Saving vectors to " + outfilename)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=100000)
    np.set_printoptions(sign=' ')
    np.set_printoptions(floatmode='fixed')
    np.set_printoptions(threshold=100000)
    
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
    args = parser.parse_args()

    vectors, ndim = read_vectors(args.infile[0])

    d = calc_vecs(vectors)
    write_vecs(d, ndim, args.outfile[0])

