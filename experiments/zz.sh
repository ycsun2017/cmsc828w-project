##### running script on lc
#!/bin/bash
#BSUB -q pbatch 
#BSUB -G guests 
#BSUB -W 720
#BSUB -nnodes 4 
#BSUB -o /g/g16/zhang65/src/sitepackages/llrl/experiments/log3.txt 

##### These are shell commands
date
#source /g/g16/zhang65/.conda/bin/activate
conda activate py37
export LD_LIBRARY_PATH=/g/g16/zhang65/.conda/pkgs/cudatoolkit-10.2.89-684.g752c550a/lib:$LD_LIBRARY_PATH

##### Launch parallel job using srun
##### 
cd /g/g16/zhang65/src/sitepackages/llrl

timestamp=$(date +%b_%d_%y_%H_%M_%s)

RUNS=10
eps=(25 50 70)
masses=(0.5 1 5 10)
for ((i=0;i<${RUNS};i++));
do
  for j in "${eps[@]}";
  do
    for k in "${masses[@]}";
    do
        jsrun -n 1 -a 1 -g 1 -c 1 python mml_zzz.py --run ${i} --meta_update_every ${j} --mass ${k}
    done
    echo $k
   done 
  echo $j
done 

# for seed in 1234 do
#     for samples in 100 250 500 1000
#         jsrun -n 1 -a 1 -g 1 -c 1 python meta.py --seed $seed --samples $samples --meta-episodes 10 --coeff 0.5 --meta_update_every 25 
#     done 
# done 

date
echo "Done"
