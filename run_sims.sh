#!/bin/bash

#python_exec="addqueue -s -q cmb -m 1 -n 1x12 /usr/bin/python3"
python_exec="python3"
#predir="/users/damonge/SO/BBMoments/sims_gauss_fullsky_nosiv_ns256_csd/"
predir="./"

seed_ini=1000
#seed_end=1500
seed_end=1001

rm -f sims_list.txt
for (( i=${seed_ini}; i<=${seed_end}; i++ ))
do
    dirname=${predir}'s'${i}
    echo ${dirname}
    echo ${dirname} >> sims_list.txt
    ${python_exec} simulation.py --output-dir ${dirname} --seed ${i} --nside 256
done

for (( i=${seed_ini}; i<=${seed_end}; i++ ))
do
    dirname=${predir}'s'${i}
    echo ${dirname}
    cp sims_list.txt ${dirname}
done
rm -f sims_list.txt
