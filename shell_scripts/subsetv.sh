#!/bin/bash
cd /g/data3/hh5/tmp/as3189/OFAM/

for f in ocean_v_21*.nc; do
	echo "$f"
	ncks -O -d yu_ocean,-15.0,14.9 -d xu_ocean,120.0,294.9 "${f}" "/g/data/e14/as3189/OFAM/trop_pac/${f}";
done
