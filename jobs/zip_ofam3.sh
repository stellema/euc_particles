#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=4:00:00
#PBS -l mem=4GB
#PBS -l ncpus=10
#PBS -l storage=gdata/e14
#PBS -l wd

##############################################
# Compress OFAM3 data files.
## Historical: for y in {198..201}
## Projection: for year in {207..201}
###############################################

cd /g/data/e14/as3189/OFAM/trop_pac
for var in "adic" "alk" "caco3" "det" "dic" "fe" "no3" "o2" "phy" "zoo"; do
  gzip ocean_"$var"_*.nc &
done
wait
