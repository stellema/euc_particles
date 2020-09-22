#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=500GB
#PBS -l ncpus=144
#PBS -l wd
#PBS -M astellemas@gmail.com
#PBS -m ae
#PBS -l storage=gdata/hh5+gdata/e14

module use /g/data3/hh5/public/modules
module load conda/analysis3-20.04
module unload openmpi
module load openmpi/4.0.2

EXP1="hist"
EXP2="rcp"
LON=190
FILE1="sim_${EXP1}_${LON}_v0r00.nc"
FILE2="sim_${EXP2}_${LON}_v0r00.nc"
mpirun -np 72 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP1 -x $LON -r 780 -v 0 -f $FILE1 &
mpirun -np 72 python3 /g/data/e14/as3189/OFAM/scripts/sim.py -e $EXP2 -x $LON -r 780 -v 0 -f $FILE2 & 
wait

