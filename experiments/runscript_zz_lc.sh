##### running script on lc
!/bin/bash
BSUB -q pbatch 
BSUB -G guests 
BSUB -W 720
BSUB -nnodes 4 
BSUB -o /g/g16/zhang65/src/sitepackages/maensemble/particle_envs/runscripts/train_log.txt 

##### These are shell commands
date
source /usr/mic/bio/anaconda3_powerai/bin/activate
conda activate powerai
export LD_LIBRARY_PATH=/usr/mic/bio/anaconda3_powerai/pkgs/cudatoolkit-dev-10.2.89-654.g0f7a43a/compat:/usr/mic/bio/anaconda3_powerai/pkgs/cudatoolkit-dev-10.2.89-654.g0f7a43
a/lib:$LD_LIBRARY_PATH

##### Launch parallel job using srun
##### 
cd /g/g16/zhang65/src/sitepackages/llrl

timestamp=$(date +%b_%d_%y_%H_%M_%s)
for seed in 1234 do
    for samples in 100 250 500 1000
        jsrun -n 1 -a 1 -g 1 -c 1 python meta.py --seed $seed --samples $samples --meta-episodes 10 --coeff 0.5 --meta_update_every 25 
    done 
done 

date
echo "Done"
