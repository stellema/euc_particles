#!/bin/bash
###############################################################################
#                                                                             #
#         Compress (gzip) physical lagrangian experiment data files.          #
#                                                                             #
###############################################################################
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=4:00:00
#PBS -l mem=1GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14

cd /g/data/e14/as3189/OFAM/data/v1
gzip --best -v plx*rcp*.nc
gzip --best -v plx*hist*190*.nc
gzip --best -v plx*hist*220*.nc
