#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=20GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
module unload openmpi
module load openmpi/4.0.2

python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "hist" -x 190 -r 0 -v 0
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "rcp" -x 190 -r 0 -v 0

