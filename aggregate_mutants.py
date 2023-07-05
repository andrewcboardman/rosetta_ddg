from tqdm import tqdm
import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',action='store',help='Input directory containing relaxed structures')
    parser.add_argument('-o','--output',action='store',help='Output directory for mutant decoys and scores')
    parser.add_argument('-m','--mutants',action='store',help='Mutants file')
    args = parser.parse_args()
    ddg_results = []
    for file in os.listdir(args.input):
        if file.endswith('.csv') and len(open(f"{args.input}/{file}",'r').readlines()) > 0:
            ddg_results.append(pd.read_csv(f"{args.input}/{file}"))

    ddg_results = pd.concat(ddg_results,axis=0)

    mutants = pd.read_csv(args.mutants)
    ddg_results = ddg_results.merge(mutants,on = 'rosetta_mutant',how = 'right')
    ddg_results.to_csv(args.output,index=False)

if __name__ == '__main__':
    main()