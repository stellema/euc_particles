#!/bin/bash

cd /g/data/e14/as3189/OFAM/hist/

for f in ocean_v_2014*.nc.gz; do
	echo "$f"
	mdss -P e14 put "${f}" "as3189/OFAM";
done
