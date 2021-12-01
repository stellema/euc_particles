#!/bin/bash
###############################################################################
#                                                                             #
#                        Uncompress OFAM3 data files.                         #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=1GB
#PBS -l ncpus=4
#PBS -l storage=gdata/e14
#PBS -l wd

cd /g/data/e14/as3189/OFAM/trop_pac

for var in "u" "v" "w"; do
  # Historical 1981-2012
  for year in {1..9}; do
    gunzip  ocean_"$var"_198"$year"*.nc.gz &
  done
  gunzip  ocean_"$var"_199*.nc.gz &
  gunzip  ocean_"$var"_200*.nc.gz &
  for year in {0..2}; do
    gunzip  ocean_"$var"_201"$year"*.nc.gz &
  done

  # RCP8.5 2070-2101
  gunzip  ocean_"$var"_21*.nc.gz &
  for year in {7..9}; do
    gunzip  ocean_"$var"_20"$year"*.nc.gz &
  done
done
wait
