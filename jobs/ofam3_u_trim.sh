#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=8GB
#PBS -l ncpus=1
#PBS -l wd
#PBS -l storage=gdata/e14

###############################################################################
# Subset lat/lon of OFAM3 "u" veocity output.
###############################################################################

module load nco
module load cdo
cd /g/data/e14/as3189/OFAM/trop_pac/tape/
for f in ocean_u_*.nc
do
echo "$f"
ncks -O -d yu_ocean,-14.9,15.0 -d xu_ocean,120.09,295.0 "${f}" "../${f}"
rm "$f"
done
