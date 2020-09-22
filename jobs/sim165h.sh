#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=380GB
#PBS -l ncpus=96
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
module unload openmpi
module load openmpi/4.0.2

EXP="hist"
LON=165
FILE="sim_${EXP}_${LON}_v0r00.nc"
python3 /g/data/e14/as3189/OFAM/scripts/sim_particleset.py -e $EXP -x $LON -r 0 -v 0
mpirun -np 96 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP -x $LON -r 1086 -v 0 -f $FILE