#!/bin/bash
#PBS -P e14
#PBS -q copyq
#PBS -l walltime=0:30:00
#PBS -l mem=1GB
#PBS -l ncpus=1
#PBS -l storage=scratch/e14+gdata/e14
#PBS -l other=mdss
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com

ls /scratch/e14/as3189/plxv1/
mdss -P e14 put /scratch/e14/as3189/plxv1/*.gz as3189/plx/v1
