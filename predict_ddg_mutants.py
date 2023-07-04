from tqdm import tqdm
from multiprocessing import Pool
import os
from functools import partial
import pandas as pd
import argparse
import utils
import json


def estimate_ddg(wt_pose_id, mutants, 
                 wt_pose_filepath='../examples/wt_relax/', 
                 mutant_pose_filepath = '../examples/mutants/', 
                 scorefxn_names = ['franklin2019','elec', 'van_der_waals','solvation'], relax_wt=False, relax_mutant=True, relax_params = None):
    # Start PyRosetta
    utils.init_pyrosetta()

    # setup score functions
    scorefxns = utils.initiate_scorefunction(scorefxn_names)

    # Default relax params
    if relax_params is None:
        relax_params = {
            'sample_level': 'chi',
            'radius': 9,
            'num_bb_nbrs': 2
        }

    wt_scores = []
    mutant_scores = []

    # Load wt structure  
    wt_pose = utils.load_pose_and_membrane(f'{wt_pose_filepath}/{wt_pose_id}.pdb')

    # Optionally relax wt structure into membrane
    if relax_wt:    
        wt_pose = utils.relax_pose(
            wt_pose, scorefxns['franklin2019'], 
            target_position=1, radius = 1000, sample_level='chi', cycles=1
        )

    # Score wild type pose  
    for score_name, scorefxn in scorefxns.items():
        E_wt = scorefxn.score(wt_pose)
        wt_scores.append({
            'decoy': wt_pose_id,
            'score_name': score_name,
            'E_wt': E_wt
        })
    with open(f'{mutant_pose_filepath}/{wt_pose_id}_wt.json', 'w') as f:
        json.dump(wt_scores, f)
    
    for mutant in tqdm(mutants): 
        # Mutate residue
        mutant_pose = utils.mutate_residue(wt_pose, scorefxns['franklin2019'], mutant=mutant)
        
        if relax_mutant:
            # Relax mutant structure
            mutant_pose = utils.relax_pose(
                mutant_pose, scorefxns['franklin2019'], 
                target_position=int(mutant[1:-1]), **relax_params, constrained=True, cycles = 1
            )

        # dump mutant structure
        mutant_pose.dump_pdb(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.pdb')

        # Score mutant pose
        mutant_scores_ = []
        for score_name, scorefxn in scorefxns.items():
            E_mut = scorefxn.score(mutant_pose)
            mutant_scores_.append({
                    'decoy': wt_pose_id,
                    'mutant': mutant,
                    'score_name': score_name,
                    'E_mut': E_mut
            })
        with open(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.json', 'w') as f:
            json.dump(mutant_scores_, f)
        for score in mutant_scores_:
            mutant_scores.append(score)
        
    mutant_scores = pd.DataFrame(mutant_scores)
    wt_scores = pd.DataFrame(wt_scores)
    mutant_scores = mutant_scores.merge(wt_scores, on=['decoy', 'score_name'])
    mutant_scores['ddg'] = mutant_scores['E_mut'] - mutant_scores['E_wt']
    return mutant_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',action='store',help='Input directory containing relaxed structures')
    parser.add_argument('-o','--output',action='store',help='Output directory for mutant decoys and scores')
    parser.add_argument('-m','--mutants',action='store',help='Mutants file')
    parser.add_argument('-j','--n_workers',action='store',type=int,help='Number of workers')
    args = parser.parse_args()

    # read relaxed structures
    decoy_ids = [file[:-4] for file in os.listdir(args.input) if file.endswith('.pdb')]

    # Load mutants
    mutants = pd.read_csv(args.mutants)
    if 'rosetta_mutant' not in mutants.columns:
        raise ValueError('Mutants file must contain a column named "rosetta_mutant"')
    mutants_list = mutants['rosetta_mutant'].tolist()

    # Mutate and estimate ddG
    # Parallelise this step over decoy inputs, not mutants
    with Pool(args.n_workers) as p:
        estimate_ddg_partial = partial(
            estimate_ddg, 
            wt_pose_filepath =args.input, mutant_pose_filepath = args.output, 
            mutants = mutants_list, relax_wt = False, relax_mutant = True
            )
        ddg_results_decoy = p.map(estimate_ddg_partial, decoy_ids)
    
    # Aggregate results and merge with input data
    ddg_results = pd.concat(ddg_results_decoy,axis=0)
    ddg_results = ddg_results.merge(mutants,on = 'rosetta_mutant',how = 'right')
    ddg_results.to_csv(f"{args.output}/ddg.csv",index=False)

if __name__ == '__main__':
    main()