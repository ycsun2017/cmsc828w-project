#! /bin/bash  
RUNS=10
for ((i=5;i<${RUNS};i++));
do
    python meta.py --run ${i} --tau 0.8 --meta_update_every 50
done