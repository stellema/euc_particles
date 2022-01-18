#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=19:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON,EXP,Y,R

###############################################################################
#
# Run PLX spinup.
# To run: qsub -v LON=250,EXP="hist",Y=6,R=10 spinup.sh
# where Y is the offset year and R is the file increment
###############################################################################

ECHO=/bin/echo
parent=/g/data/e14/as3189/stellema/plx/

# Run spinup particles
if [ $R -lt 13 ]; then
  $ECHO "Run plx spinup for $EXP at lon $LON."
  mpirun -np $PBS_NCPUS python3 "$parent"/scripts/plx_spinup.py -e $EXP -x $LON -v 1 -r 5 -y $Y

  # Submit job to create particle set restart file.
  if [ $R -lt 12 ]; then
    R=$R + 1
    qsub -v $LON,EXP=$EXP,Y=$Y,R=$R "$parent"/jobs/spinup_ps.sh
  fi
fi
