#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=8:00:00
#PBS -l mem=3GB
#PBS -l ncpus=3
#PBS -l storage=gdata/e14
#PBS -l wd
cd /g/data/e14/as3189/OFAM/trop_pac
gzip -9 ocean_w_1*.nc &
gzip -9 ocean_v_1*.nc &
gzip -9 ocean_u_1*.nc &
wait

