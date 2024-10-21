#!/bin/bash
#SBATCH --job-name=gridsearch_k3         # Job name
#SBATCH --output=k3-job-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --cpus-per-task=50            # 50 CPU cores for parallelism
#SBATCH --mem=32GB                    # Total memory for the job (adjust based on need)
#SBATCH --time=12:00:00               # Time limit for the job (e.g., 2 hours)

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.8.16-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/pattern-rec/bin/activate

############ GETTING PRE-TRAINED PRUNED MODEL:

# make a directory in the TMPDIR

# copy code into TMPDIR
cp -r /scratch/$USER/pattern-rec/Pattern-Recognition-Gr7/Assignment-2 $TMPDIR

mkdir -p $TMPDIR/Assignment-2/Data-PR-As2/Genes/

# copy data into TMPDIR data folder
# /Assignment-2/Data-PR-As2/Genes
cp -r /scratch/$USER/pattern-rec/genes/* $TMPDIR/Assignment-2/Data-PR-As2/Genes

tree $TMPDIR

############ RUN CODE:

# Navigate to TMPDIR
cd $TMPDIR/Assignment-2

# Run training
python3 run_gridsearch.py --cv KFold_3

############ SAVING:

# Save models by compressing and copying from TMPDIR
tar czvf /scratch/$USER/pattern-rec/results/KFold_3_gridsearch_results.tar.gz $TMPDIR/Assignment-2/gridsearch_results