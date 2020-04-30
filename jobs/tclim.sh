#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=15:00:00
#PBS -l mem=30GB
#PBS -l ncpus=2
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m abe
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.01
for i in 4; do
	for j in 3 4; do
    	python3 /g/data/e14/as3189/OFAM/scripts/create_file_ofam_climo.py $i $j &
	done
done

wait

