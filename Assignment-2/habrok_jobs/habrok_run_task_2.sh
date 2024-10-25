#!/bin/bash
#SBATCH --job-name=attempt1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16GB                                            

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.9.6-GCCcore-11.2.0   
 
# activate virtual environment
source $HOME/venvs/first_enviro/bin/activate

# # make a directory in the TMPDIR for the pre-trained model
mkdir $TMPDIR/pt


# # extract pre-trained model from scratch to TMPDIR/pt
# # Change 'try' to match the folder containing the desired model
#tar xzf /scratch/$USER/cityscapes/FAMO/results.tar.gz -C $TMPDIR/pt

# Copy code to $TMPDIR
cp -r /scratch/$USER/Assignment-2 $TMPDIR


# Navigate to TMPDIR
cd $TMPDIR/Assignment-2

# make results directory
mkdir $TMPDIR/Assignment-2/results_task2


# Run training
# mkdir -p ./save
# mkdir -p ./trainlogs
echo "Run code..."
python -u  semi-supervised.py

# Save models by compressing and copying from TMPDIR
mkdir -p /scratch/$USER/task2_results/job_${SLURM_JOBID}
# cp $TMPDIR/results /scratch/$USER/trained_FAMO_model/job_${SLURM_JOBID}
# tar czvf /scratch/$USER/trained_FAMO_model/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results

echo "Moving results to $scratch_dir..."
mv $TMPDIR/Assignment-2/results_task2 /scratch/$USER/task2_results/job_${SLURM_JOBID}
cp /scratch/$USER/Assignment-2/slurm-${SLURM_JOBID}.out /scratch/$USER/task2_results/job_${SLURM_JOBID}/results_task2
echo "Training completed and results moved successfully."