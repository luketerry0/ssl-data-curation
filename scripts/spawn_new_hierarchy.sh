#!/bin/bash
# this bash script will create a new hierarchical cluster in ourdisk and begin the jobs based on the config file
# it will also make a job to iterate through the clusters and create some metrics based on them

echo "spawning new hierarchy..."

# change working directory to scripts folder
cd $HOME/ssl-data-curation/scripts

# if the parameter is not passed, throw an error (instead of deleting anything in ourdisk :( )
if [ "${1}" = "" ]; then
  echo "pass a config name!"
  exit 1  # Exit with a non-zero exit code to indicate an error
fi

# if the output directory exists, delete it to make way for the hierarchy
if [ -d "/ourdisk/hpc/ai2es/luketerry/${1}" ]; then
  rm -rf "/ourdisk/hpc/ai2es/luketerry/${1}"
fi

# using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/luketerry/miniforge3/envs/ssl-data-curation

# call the hierarchical clustering code to create the cluster
python hierarchical_kmeans_launcher.py \
   --exp_dir /ourdisk/hpc/ai2es/luketerry/${1} \
   --embeddings_path=/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_shard_0.npy \
   --config_file ../configs/${1}.yaml 

# call the script within the newly created hierarchy to queue up the jobs
FINAL_ID=$(bash /ourdisk/hpc/ai2es/luketerry/${1}/OSCER_launcher.sh | tail -1 | grep -oE '[^ ]+$')
echo $FINAL_ID

# call the PIL data viz code to produce a cluster visualization when we're done making the hierarchy
# sbatch --dependency afterok:${FINAL_ID} $HOME/PIL_data_vis/viz.slurm ${1}
