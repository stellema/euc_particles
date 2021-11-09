#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=0:30:00
#PBS -l mem=1GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14
#PBS -l other=mdss
#PBS -l wd
#PBS -m e
#PBS -M astellemas@gmail.com
cd /g/data/e14/as3189/stellema/plx/data/v1
gzip -v -k -9 plx_hist_250_v1r09.nc
mdss -P e14 put /g/data/e14/as3189/stellema/plx/data/v1/plx_hist_250_v1r09.nc.gz as3189/plx/v1
