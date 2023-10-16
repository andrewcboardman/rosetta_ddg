import utils
import json
import argparse
from functools import partial
from multiprocessing import Pool
import os
import pandas as pd
import shutil


def relax_wt(input_structure, output_path, index, 
             scorefxn = 'franklin2019', sample_level = 'chi', cycles = 1, constrained=True, ramp_constraints=False, cartesian = False, ):
    
    input_name = os.path.basename(input_structure).split('.')[0]

    if os.path.exists(f'{output_path}/{input_name}_{index}.pdb'):
        utils.init_pyrosetta()
        scorefxns = utils.initiate_scorefunction([scorefxn])    
        wt_pose = utils.load_pose_and_membrane(f'{output_path}/{input_name}_{index}.pdb')
        E_wt = scorefxns[scorefxn].score(wt_pose)
        return E_wt
    else:
        utils.init_pyrosetta()
        scorefxns = utils.initiate_scorefunction([scorefxn],cartesian=cartesian)    
        wt_pose = utils.load_pose_and_membrane(input_structure)

        wt_pose = utils.relax_pose(
                wt_pose, scorefxns[scorefxn], 
                target_position=None, sample_level=sample_level, cycles=cycles, cartesian=cartesian, constrained=constrained, ramp_constraints=ramp_constraints
            )
        
        E_wt = scorefxns[scorefxn].score(wt_pose)
        
        wt_pose.dump_pdb(f'{output_path}/{input_name}_{index}.pdb')
        return E_wt 

    
def filter_relaxed_structures(E_wt, iter, input_structure, structures_path, n_select=10):
    """Filter relaxed structures and select top n templates"""
    df = pd.DataFrame()
    df['E_wt'] = E_wt
    df['iter'] = iter
    df.sort_values('E_wt',inplace=True)
    df.to_csv(f'{structures_path}/relax_wt_scores.csv',index=False)
    if n_select <= len(iter):
        df_select = df.head(n_select)
        selected_structures = df_select['iter'].tolist()
    if not os.path.exists(f'{structures_path}/select'):
        os.mkdir(f'{structures_path}/select')
    
    input_name = os.path.basename(input_structure).split('.')[0]
    for i in selected_structures:
        shutil.copyfile(
            f'{structures_path}/{input_name}_{i}.pdb',
            f'{structures_path}/select/{input_name}_{i}.pdb'
            )
    df_select.to_csv(f'{structures_path}/select/relax_wt_scores.csv',index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Relax a wild-type structure')
    parser.add_argument('-i', '--input_structure', help='PDB file containing the structure to relax')
    parser.add_argument('-o', '--output_path', default='./relax', help='Output file name')
    parser.add_argument('-N', '--n_struct', default=1, type=int, help='Number of structures to generate')
    parser.add_argument('-n', '--n_select', default=1, type=int, help='Number of structures to select')
    parser.add_argument('-p', '--params', default=None, type = str, help='JSON file containing relaxation parameters')
    parser.add_argument('-j','--num_cores', default=1, type=int, help='Number of processes to use for parallelization')
    args = parser.parse_args()

    if args.params is None:
        relax_params = {
            'scorefxn':'franklin2019',
            'sample_level':'bb',
            'cycles':3,
            'constrained':True,
            'ramp_constraints':False,
            'cartesian':True
        }
    else:
        relax_params = json.load(open(args.params,'r'))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    relax_wt_partial = partial(relax_wt, args.input_structure, args.output_path, 
                       scorefxn=relax_params['scorefxn'], 
                       sample_level = relax_params['sample_level'], 
                       cycles = relax_params['cycles'], 
                       constrained=relax_params['constrained'], 
                       ramp_constraints=relax_params['ramp_constraints'],
                       cartesian = relax_params['cartesian'])
    
    iter = range(args.n_struct)
    
    with Pool(args.num_cores) as p:
        E_wt = p.map(relax_wt_partial, iter)

    filter_relaxed_structures(E_wt, iter,args.input_structure, args.output_path, n_select=args.n_select)
    