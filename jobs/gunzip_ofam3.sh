#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=1GB
#PBS -l ncpus=4
#PBS -l storage=gdata/e14
#PBS -l wd

#=============================================================================
#
# Uncompress OFAM3 data files.
#
#=============================================================================

cd /g/data/e14/as3189/OFAM/trop_pac

# Historical 1981-2012
for var in "u" "v" "w"; do
  for y in {1981..2012}; do
    for m in {01..12}; do
      gunzip -v ocean_"$var"_"$y"_"$m".nc.gz &
    done
  done
done

# RCP8.5 2070-2101
for var in "u" "v" "w"; do
  for y in {2070..2101}; do
    for m in {01..12}; do
      gunzip -v ocean_"$var"_"$y"_"$m".nc.gz &
    done
  done
done
wait
