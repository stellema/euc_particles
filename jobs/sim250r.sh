#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
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

EXP="rcp"
LON=250
FILE="sim_${EXP}_${LON}_v0r00.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e $EXP -x $LON -r 0 -v 0
mpirun python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP -x $LON -r 780 -v 0 -f $FILE
