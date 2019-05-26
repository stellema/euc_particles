#!/bin/bash

module load cdo
module load nco

# Data folder.
folder = "/g/data/e14/as3189/OFAM/OFAM3_BGC_SPINUP_03/daily/"

# Move data to this folder.
new_folder = "/g/data/e14/as3189/OFAM/temp/"

#for var in "u" "v" "w" "salinity" "temp";
for var in "u";
do
    for year in 2010;
    do
        file = "ocean_${var}_${year}";

        # TODO: Subset data. (need monthly for loop)

        # Merge into annual file (will be deleted later).
        cdo mergetime "${folder}${file}_*.nc" "${new_folder}${file}_merged.nc";

        # 5-daily average.
        cdo timselmean,5 "${new_folder}${file}_merged.nc" "${new_folder}${file}_5-day.nc";

        # Delete merged file.
        rm "${new_folder}${file}_merged.nc";

        ## Optional:
        ## Split back into monthly files.
        #cdo splitmon "${new_folder}${file}_5-day.nc" "${new_folder}${file}_5-day";

        ## Delete annual file.
        #rm "${new_folder}${file}_5-day.nc";



    done;
done;
