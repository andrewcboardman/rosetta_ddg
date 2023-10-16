from tqdm import tqdm
from multiprocessing import Pool
import os
from functools import partial
from itertools import product
import pandas as pd
import argparse
import utils
import json
from pyrosetta.toolbox.mutants import mutate_residue




def estimate_ddg(wt_pose_id, mutant, 
                 wt_pose_filepath='../examples/wt_relax/', 
                 mutant_pose_filepath = '../examples/mutants/', 
                 scorefxn_names = ['franklin2019','elec', 'van_der_waals','solvation'], relax_wt=False, relax_mutant=True, relax_params = None):
    # Start PyRosetta
    if not os.path.exists(mutant_pose_filepath):
        os.mkdir(mutant_pose_filepath)

    if os.path.exists(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.csv') and (os.stat(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.csv').st_size != 0):
        return pd.read_csv(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.csv')        
            
    else:
        # Create placeholder file to prevent other processes working on this mutant
        open(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.csv', 'a').close()
        utils.init_pyrosetta()
        
        # Default relax params
        if relax_params is None:
            relax_params = {
            'sample_level':'chi',
            'cycles':1,
            'constrained':True,
            'ramp_constraints':False,
            'cartesian':True
        }

        # setup score functions
        scorefxns = utils.initiate_scorefunction(scorefxn_names, cartesian=relax_params['cartesian'])

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
    
        # Mutate residue
        pos = int(mutant[1:-1])
        mut_aa = mutant[-1]
        mutant_pose =  wt_pose.clone()
        mutate_residue(mutant_pose, pos, mut_aa, pack_scorefxn=scorefxns['franklin2019'], pack_radius=5.0)
        
        if relax_mutant:
            # Relax mutant structure
            mutant_pose = utils.relax_pose(
                mutant_pose, scorefxns['franklin2019'], 
                target_position=int(mutant[1:-1]), **relax_params,
            )

        # dump mutant structure
        mutant_pose.dump_pdb(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.pdb')

        # Score mutant pose
        mutant_scores = []
        for score_name, scorefxn in scorefxns.items():
            E_mut = scorefxn.score(mutant_pose)
            mutant_scores.append({
                    'decoy': wt_pose_id,
                    'rosetta_mutant': mutant,
                    'score_name': score_name,
                    'E_mut': E_mut
            })

        mutant_scores = pd.DataFrame(mutant_scores)
        wt_scores = pd.DataFrame(wt_scores)
        mutant_scores = mutant_scores.merge(wt_scores, on=['decoy', 'score_name'])
        mutant_scores['ddg'] = mutant_scores['E_mut'] - mutant_scores['E_wt']
        mutant_scores.to_csv(f'{mutant_pose_filepath}/{wt_pose_id}_{mutant}.csv',index=False)
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
            relax_wt = False, relax_mutant = True
            )
        ddg_results_decoy = p.starmap(estimate_ddg_partial, product(decoy_ids, mutants_list))
    
        # Aggregate results and merge with input data

    ddg_results = pd.concat(ddg_results_decoy,axis=0)
    ddg_results = ddg_results.merge(mutants,on = 'rosetta_mutant',how = 'right')
    ddg_results.to_csv(f"{args.output}/ddg.csv",index=False)

if __name__ == '__main__':
    main()
