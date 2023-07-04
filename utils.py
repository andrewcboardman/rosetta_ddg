
import pyrosetta as pr
from pyrosetta.rosetta.core.select import residue_selector
from pyrosetta.rosetta.protocols.membrane import AddMembraneMover
from pyrosetta.rosetta.core.pack.task import operation, TaskFactory
from pyrosetta.rosetta.utility import vector1_bool
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover


def check_valid_mutant(pose, mutant):
    pos = int(mutant[1:-1])
    if (pose.total_residue() < pos) or (pos < 1):
        return False
    elif not (mutant[-1] in 'ACDEFGHIKLMNPQRSTVWY'):
        return False
    elif not pose.residue(pos).name1() == mutant[0]:
        return False
    else:
        return True


# Utility functions
def init_pyrosetta():
    pr.init(
        '-use_input_sc -ignore_unrecognized_res \
        -ignore_zero_occupancy false \
        -load_PDB_components false \
        -no_fconfig',
        extra_options="-mp:lipids:has_pore false -mp:lipids:temperature 37.0 -mp:lipids:composition DLPC -ex1 -ex2 -ex1aro -ex2aro -packing:no_optH false -packing:flip_HNQ -fa_max_dis 9.0"
    )


def load_pose_and_membrane(input_pdb_path,membrane=True):
    pose = pr.pose_from_pdb(input_pdb_path)
    if membrane:
        add_memb = AddMembraneMover("from_structure")
        add_memb.apply(pose)
    return pose


def initiate_scorefunction(scorefxn_names, cartesian=False):
    # set up score functions
    scorefxns_ = {
        'franklin2019':pr.create_score_function('franklin2019'),
        'van_der_waals':pr.ScoreFunction(),
        'solvation':pr.ScoreFunction(),
        'elec':pr.ScoreFunction(),
        'hbond':pr.ScoreFunction(),
    }

    scorefxns_['franklin2019'].set_weight(pr.rosetta.core.scoring.membrane_span_constraint, 1)
    scorefxns_['franklin2019'].set_weight(pr.rosetta.core.scoring.fa_water_to_bilayer, 1.5)
    scorefxns_['van_der_waals'].set_weight(pr.rosetta.core.scoring.fa_atr,1)
    scorefxns_['van_der_waals'].set_weight(pr.rosetta.core.scoring.fa_rep,0.55)
    scorefxns_['solvation'].set_weight(pr.rosetta.core.scoring.fa_sol,1)
    scorefxns_['elec'].set_weight(pr.rosetta.core.scoring.fa_elec,1)
    scorefxns_['hbond'].set_weight(pr.rosetta.core.scoring.hbond_sr_bb,1)
    scorefxns_['hbond'].set_weight(pr.rosetta.core.scoring.hbond_lr_bb,1)
    scorefxns_['hbond'].set_weight(pr.rosetta.core.scoring.hbond_bb_sc,1)
    scorefxns_['hbond'].set_weight(pr.rosetta.core.scoring.hbond_sc,1)

    if cartesian:
        scorefxns_['franklin2019'].set_weight(pr.rosetta.core.scoring.cart_bonded, 0.5)
        scorefxns_['franklin2019'].set_weight(pr.rosetta.core.scoring.pro_close, 0.0)
    
    scorefxns = {name:scorefxns_[name] for name in scorefxn_names}
    return scorefxns



def relax_pose(pose, sfxn, target_position = None, sample_level = 'chi', cycles = 1, radius = 5, num_bb_nbrs = 2, cartesian=False, constrained = False, max_iter=None):
    # Clone wild type pose
    working_pose = pose.clone()
    working_pose.update_residue_neighbors()

    # Get level of relax
    if sample_level == 'chi':
        sample_chi = True
        sample_bb = False
    if sample_level == 'bb':
        sample_chi = True
        sample_bb = True

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    tf.push_back(operation.NoRepackDisulfides())

    if target_position is not None:
        # Select mutant and its sequence neighbors (repack backbone) and structure neighbours (repack sidechains)
        mutant_res = residue_selector.ResidueIndexSelector(target_position)
        bb_neighbours = residue_selector.PrimarySequenceNeighborhoodSelector(num_bb_nbrs, num_bb_nbrs, mutant_res, False) 
        sc_neighbours = residue_selector.NeighborhoodResidueSelector(
            mutant_res, radius, True
            )        
        sc_neighbours = residue_selector.OrResidueSelector(sc_neighbours, bb_neighbours)

        # Generate a MoveMap
        movemap = pr.rosetta.core.select.movemap.MoveMapFactory()
        if sample_chi:
            movemap.add_chi_action(pr.rosetta.core.select.movemap.mm_enable, sc_neighbours)
        if sample_bb:
            movemap.add_bb_action(pr.rosetta.core.select.movemap.mm_enable, bb_neighbours)
        movemap.all_jumps(False)

        # allow selected residues only repacking (=switch off design)
        restrict_repacking_rlt = operation.RestrictToRepackingRLT()
        restrict_subset_repacking = operation.OperateOnResidueSubset(
            restrict_repacking_rlt, sc_neighbours, False)
        tf.push_back(restrict_subset_repacking)

        #prevent all residues except selected from design and repacking
        prevent_repacking_rlt = operation.PreventRepackingRLT()
        prevent_subset_repacking = operation.OperateOnResidueSubset(
            prevent_repacking_rlt, sc_neighbours, True)
        tf.push_back(prevent_subset_repacking)
    else:
        movemap = pr.rosetta.core.select.movemap.MoveMapFactory()        
        movemap.all_chi(sample_chi)
        movemap.all_bb(sample_bb)
        movemap.all_jumps(False)

        restrict_to_repacking = operation.RestrictToRepacking()
        tf.push_back(restrict_to_repacking)

    # Perform a FastRelax 
    fastrelax = pr.rosetta.protocols.relax.FastRelax(sfxn, cycles)
    if constrained:
        fastrelax.constrain_relax_to_start_coords(True)    
    if cartesian:
        fastrelax.cartesian(True)
    if max_iter:
        fastrelax.max_iter(max_iter)
        
    fastrelax.set_task_factory(tf)
    fastrelax.set_movemap_factory(movemap)
    fastrelax.set_movemap_disables_packing_of_fixed_chi_positions(True)
    tf.create_task_and_apply_taskoperations(working_pose)
    fastrelax.apply(working_pose)
    return working_pose


def mutate_residue(pose, scorefxn, mutant):
    """Mutate a residue in a pose"""

    working_pose = pose.clone()

    if not check_valid_mutant(pose, mutant):
        raise ValueError(f'Invalid mutation {mutant}')
    else:
        # Parse mutant
        position = int(mutant[1:-1])
        mut_aa = mutant[-1]

    task = TaskFactory.create_packer_task(pose)

    # set mutant amino acid
    aa_bool = vector1_bool()
    mut_aa = aa_from_oneletter_code(mut_aa)
    for i in range(1, 21):
        aa_bool.append(i == mut_aa)

    # prevent other residue from repacking
    for i in range(1, pose.total_residue() + 1):
        if i == position:
            task.nonconst_residue_task(i).restrict_absent_canonical_aas(aa_bool)
        if i != position:
            task.nonconst_residue_task(i).prevent_repacking()

    # execute packer task
    packer = PackRotamersMover(scorefxn , task)
    packer.apply(working_pose)

    return working_pose