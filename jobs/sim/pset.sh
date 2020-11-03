#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=44GB
#PBS -l ncpus=5
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.10

python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "rcp" -x 190 -r 696 -v 0 &
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "hist" -x 220 -r 780 -v 0 &
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "rcp" -x 220 -r 780 -v 0 &
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "hist" -x 250 -r 780 -v 0 &
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e "rcp" -x 250 -r 780 -v 0 &
wait
