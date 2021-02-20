#!/bin/bash
module load cdo
for file in tau*climo.nc; do
echo "$file"
cdo -f nc -lec,0  -remapcon,$file -topo landmask.nc
cdo remapbil,grid_05 -div $file landmask.nc regrid/${file%%.*}_regrid.nc
rm landmask.nc
done
