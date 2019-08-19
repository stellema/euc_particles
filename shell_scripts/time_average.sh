#!/bin/bash

module load cdo
module load nco

# Data path.
path = "/g/data/e14/as3189/OFAM/OFAM3_BGC_SPINUP_03/daily/"

# Move data to this path.
new_path = "/g/data/e14/as3189/OFAM/temp/"

# Five-daily mean
for var in "u" "v" "w" "salt" "temp";
do
    for year in 2010;
    do
        file = "ocean_${var}_${year}";
        # Merge into annual file (will be deleted later).
        cdo mergetime "${path}${file}_*.nc" "${new_path}${file}_merged.nc";

        # 5-daily average.
        cdo timselmean,5 "${new_path}${file}_merged.nc" "${new_path}${file}_5-day.nc";

        # Delete merged file.
        rm "${new_path}${file}_merged.nc";

        ## Optional:
        ## Split back into monthly files.
        #cdo splitmon "${new_path}${file}_5-day.nc" "${new_path}${file}_5-day";

        ## Delete annual file.
        #rm "${new_path}${file}_5-day.nc";

    done;
done;
