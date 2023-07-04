import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1
import argparse


def generate_mutants(structure_file, chain = 'A', ala_leu_only=True):
    """Generate a list of all possible mutants for a given structure. Optionally restrict to just alanine/leucine mutations."""
    parser = PDB.PDBParser()
    struct = parser.get_structure('temp',structure_file)
    # Get all residues
    residues = struct[0][chain].get_residues()
    # Filter non-canonical residues
    residues = [res for res in residues if res.id[0] == ' ']
    # Extract positions and amino 
    residues = [(res.id[1], seq1(res.resname)) for res in residues]
    # Transform into a dataframe
    residues = pd.DataFrame(residues, columns=['pdb_pos', 'wt_aa'])
    # Get Rosetta numbering (1-indexed)
    residues['rosetta_pos'] = residues.index + 1
    # Generate all possible mutations
    mut_aa = list('ACDEFGHIKLMNPQRSTVWY')
    residues['mut_aa'] = residues['wt_aa'].apply(lambda x: [aa for aa in mut_aa if aa != x])
    mutants = residues.explode('mut_aa')
    # Generate a unique identifier for each mutant
    mutants['pdb_mutant'] = mutants['wt_aa'] + mutants['pdb_pos'].astype(str) + mutants['mut_aa']
    mutants['rosetta_mutant'] = mutants['wt_aa'] + mutants['rosetta_pos'].astype(str) + mutants['mut_aa']
    # Filter to alanine/leucine mutations
    if ala_leu_only:
        ala_mutants = mutants[mutants['mut_aa'] == 'A']
        ala_leu_mutants = mutants[(mutants['wt_aa'] == 'A') & (mutants['mut_aa'] == 'L')]
        mutants = pd.concat([ala_mutants, ala_leu_mutants]).sort_values('rosetta_pos')
    return mutants

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a list of all possible mutants for a given structure. Optionally restrict to just alanine/leucine mutations.')
    parser.add_argument('-i', '--input_structure_file', help='PDB file containing the structure to mutate')
    parser.add_argument('-c', '--chain', default='A', help='Chain to mutate')
    parser.add_argument('-a', '--ala_leu_only', action='store_true', help='Only generate alanine/leucine mutations')
    parser.add_argument('-o', '--output', default='mutants.csv', help='Output file name')
    args = parser.parse_args()
    mutants = generate_mutants(args.input_structure_file, args.chain, args.ala_leu_only)
    mutants.to_csv(args.output, index=False)