#!/bin/bash
#
## Uncompress OFAM3 data files.
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=1GB
#PBS -l ncpus=3
#PBS -l storage=gdata/e14
#PBS -l wd

##############################################
#
## Historical: for y in {198..201}
## Projection: for year in {207..201}
## Historical: for year in 1979 1980 2013 2014
#
###############################################
cd /g/data/e14/as3189/OFAM/trop_pac
gzip --best ocean_*climo.nc &
for var in "u" "v" "w"; do
  # Historical 1981-2012
  for year in {1..9}; do
    gunzip  ocean_"$var"_198"$year"*.nc &
  done
  gunzip  ocean_"$var"_199*.nc &
  gunzip  ocean_"$var"_200*.nc &
  for year in {0..2}; do
    gunzip  ocean_"$var"_201"$year"*.nc &
  done

  # RCP8.5 2070-2101
  gunzip  ocean_"$var"_21*.nc &
  for year in {7..9}; do
    gunzip  ocean_"$var"_20"$year"*.nc &
  done
done
wait
