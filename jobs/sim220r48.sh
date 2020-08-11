#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=190GB
#PBS -l ncpus=48
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
module unload openmpi
module load openmpi/4.0.2

EXP="rcp"
FILE1="sim_rcp_220_v0r7.nc"
FILE2="sim_rcp_220_v0r8.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e $EXP -f $FILE1
mpirun python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP -x 220 -r 732 -v 0 -f $FILE1
python3 /g/data/e14/as3189/OFAM/scripts/sim_info.py -f $FILE2