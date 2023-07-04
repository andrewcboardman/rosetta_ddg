PIPELINE_DIR=/home/andrew/Packages/rosetta_membrane_pipeline/

WT_TEMPLATE_FILE=${PIPELINE_DIR}/examples/wt_relax/wt_pose_0.pdb
WT_DECOYS_DIR=${PIPELINE_DIR}/examples/wt_relax/

N_RELAX_DECOYS=100
N_RELAX_SELECT=10

N_THREADS=5

python $PIPELINE_DIR/relax.py -i $WT_TEMPLATE_FILE -o $WT_DECOYS_DIR -N $N_RELAX_DECOYS -n $N_RELAX_SELECT -j $N_THREADS

python $PIPELINE_DIR/generate_mutants.py -i $WT_TEMPLATE_FILE -o $MUTANTS_FILE -a

python $PIPELINE_DIR/predict_ddg_mutants.py -i $WT_DECOYS_DIR/select -m $MUTANTS_FILE -o $MUTANTS_DIR -j $N_THREADS