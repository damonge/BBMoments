#!/bin/bash

python_exec="/usr/bin/python3"

nside=256
for seed in {1000..1010}
do
    for lmax in 383 #500 767
    do
	if [ ! -f data/w3j_lmax${lmax}.npz ] ; then
	    echo "Generating w3j with lmax =  "${lmax}
	    python3 utils.py --lmax ${lmax}
	fi
	#if [ ! -d simulations_only_dust/sims_ns${nside}_seed${seed}_lm${lmax} ] ; then
	    echo "Generating simulation "${seed}
	   python3 example0.py --nside ${nside} --seed ${seed} --lmax ${lmax}
	#fi
    done
done

