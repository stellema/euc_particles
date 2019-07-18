#!/bin/bash
cd /g/data/e14/as3189/OFAM/hist/

for f in ocean_v*.nc; do
	echo "$f"
	ncks -O -d yu_ocean,-15.0,14.9 -d xu_ocean,120.0,294.9 "${f}" "../trop_pac/${f}";
done
