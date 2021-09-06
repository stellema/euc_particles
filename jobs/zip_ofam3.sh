#!/bin/bash
#
## Compress OFAM3 data files.
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=3GB
#PBS -l ncpus=3
#PBS -l storage=gdata/e14
#PBS -l wd

cd /g/data/e14/as3189/OFAM/trop_pac
for var in "u" "v" "w"
do
  for year in 207 208 209 210
  do
    gzip --best ocean_"$var"_"$year"*.nc &
  done
done
wait

