#!/bin/bash
#
## Compress OFAM3 data files.
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
  for year in 1980 197 2013 2014; do
    gzip --best ocean_"$var"_"$year"*.nc &
  done
done
wait

