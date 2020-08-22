#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=8:30:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
module unload openmpi
module load openmpi/4.0.2

EXP="hist"
mpirun python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP -x 165 -r 1296 -v 5
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f "sim_hist_165_v5r0.nc"
