#! /bin/bash  
RUNS=10
for ((i=0;i<${RUNS};i++));
do
    python meta.py --run ${i} --tau 0.8 --meta_update_every 75
done