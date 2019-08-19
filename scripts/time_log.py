# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 04:59:17 2019

@author: Annette Stellema
qsub -I -l walltime=6:00:00,ncpus=3,mem=50GB -P e14 -q

_______________________________________________________________________________

Test 0:
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00

Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 1 [165]
Particles (/repeatdt): 42
Particles (total): 2520
Time decorator used. 

ofam_fieldset: 0 hours, 0 mins, 00.85 secs (03:43am)

Executing: ParticleFile_1979-1979_v0i
EUC_pset: 0 hours, 3 mins, 10.67 secs (03:46am)
Particles removed: 19
remove_westward_particles: 0 hours, 0 mins, 00.00 secs (03:46am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==> 
100% (31449600.0 of 31449600.0) |########| Elapsed Time: 0:25:44 Time:  0:25:44
Particles removed: 0
remove_westward_particles: 0 hours, 0 mins, 00.02 secs (04:12am)
EUC_particles: 0 hours, 29 mins, 08.66 secs (04:12am)

Timer (base): 0 hours, 29 mins, 15.85 secs



Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00

Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 1 [165]
Particles (/repeatdt): 42
Particles (total): 2520
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 00.79 secs (01:15am)
Executing: ParticleFile_1979-1979_v0i
EUC_pset: 0 hours, 3 mins, 15.10 secs (01:18am)
Particles removed: 19
remove_westward_particles: 0 hours, 0 mins, 00.00 secs (01:18am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==> 

_______________________________________________________________________________

Test 1 (66.66% increase in particles; 64.74% increase in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00

Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 3 [165 190 200]
Particles (/repeatdt): 126
Particles (total): 7560
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 00.79 secs (04:35am)

Executing: ParticleFile_1979-1979_v1i
EUC_pset: 0 hours, 9 mins, 27.63 secs (04:45am)
Particles removed: 61
remove_westward_particles: 0 hours, 0 mins, 00.03 secs (04:45am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==>
100% (31449600.0 of 31449600.0) |########| Elapsed Time: 0:39:14 Time:  0:39:14
Particles removed: 0
remove_westward_particles: 0 hours, 0 mins, 00.06 secs (05:24am)
EUC_particles: 0 hours, 48 mins, 50.37 secs (05:24am)

Particles removed (final): 2946
ParticleFile_transport: 0 hours, 0 mins, 06.74 secs (05:24am)

Timer (base): 0 hours, 48 mins, 12.55 secs
_______________________________________________________________________________

Test 2 (doubled dt; 1.66% decrease in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00

Runtime: 364 days
Timestep (dt): 240 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 1 [165]
Particles (/repeatdt): 42
Particles (total): 2520
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 00.80 secs (06:40am)
Executing: ParticleFile_1979-1979_v2i
EUC_pset: 0 hours, 3 mins, 12.89 secs (06:43am)
Particles removed: 19
remove_westward_particles: 0 hours, 0 mins, 00.00 secs (06:43am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==>
100% (31449600.0 of 31449600.0) |########| Elapsed Time: 0:25:25 Time:  0:25:25
EUC_particles: 0 hours, 28 mins, 45.85 secs (07:09am)

Particles removed (final): 812
ParticleFile_transport: 0 hours, 0 mins, 04.00 secs (07:09am)

Timer (base): 0 hours, 28 mins, 50.65 secs
_______________________________________________________________________________

Test 3 (half runtime; 56.87% decrease in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-06-30 00:00:00

Runtime: 180 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 1 [165]
Particles (/repeatdt): 42
Particles (total): 1260
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 00.47 secs (07:14am)
Executing: ParticleFile_1979-1979_v3i
EUC_pset: 0 hours, 3 mins, 13.86 secs (07:17am)
Particles removed: 8
remove_westward_particles: 0 hours, 0 mins, 00.00 secs (07:17am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==>
100% (15552000.0 of 15552000.0) |########| Elapsed Time: 0:09:14 Time:  0:09:14

EUC_particles: 0 hours, 12 mins, 36.80 secs (07:27am)
Particles removed (final): 461
ParticleFile_transport: 0 hours, 0 mins, 01.09 secs (07:27am)

Timer (base): 0 hours, 12 mins, 38.36 secs
_______________________________________________________________________________

Test 4 (double runtime; 147% increase in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1980-12-31 00:00:00

Runtime: 730 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: #1 [165]
Particles (/repeatdt): 42
Particles (total): 5082
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 01.98 secs (07:42am)
Executing: ParticleFile_1979-1980_v0i
EUC_pset: 0 hours, 3 mins, 10.03 secs (07:45am)
Particles removed: 20
remove_westward_particles: 0 hours, 0 mins, 00.00 secs (07:45am)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==> 
100% (63072000.0 of 63072000.0) |########| Elapsed Time: 1:08:49 Time:  1:08:49
EUC_particles: 1 hours, 12 mins, 07.96 secs (08:54am)
Particles removed (final): 1689
ParticleFile_transport: 0 hours, 0 mins, 12.88 secs (08:54am)
Timer (base): 1 hours, 12 mins, 22.82 secs

_______________________________________________________________________________

Test 5 (23x more particles; 871% increase in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00
Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 12 dz=25 [25 to 300]
Latitudes: 27 dy=0.2 [-2.6 to 2.6]
Longitudes: 3 [165 190 200]
Particles (/repeatdt): 972
Particles (total): 58320
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 00.85 secs (04:58pm)
Executing: ParticleFile_1979-1979_v4i

EUC_pset: 1 hours, 11 mins, 10.03 secs (06:13pm) (estimate)
-------
EUC_particles: 4 hours, 43 mins, 41.85 secs (09:41pm)
Particles removed (final): 23491
ParticleFile_transport: 0 hours, 0 mins, 35.53 secs (09:42pm)
Timer (base): 4 hours, 44 mins, 18.23 secs

_______________________________________________________________________________

Test 6 (2x more particles; % increase in time):
_______________________________________________________________________________
Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00
Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 14 dy=0.4 [-2.6 to 2.6]
Longitudes: 1 [165]
Particles (/repeatdt): 84
Particles (total): 5040
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 07.02 secs (10:30pm)
Executing: ParticleFile_1979-1979_v6i
EUC_pset: 0 hours, 6 mins, 32.90 secs (10:37pm)
Particles removed: 44
remove_westward_particles: 0 hours, 0 mins, 00.02 secs (10:37pm)
INFO: Compiled tparticleDeleteWestwardAgeAdvectionRK4_3D ==> 
100% (31449600.0 of 31449600.0) |#################| Elapsed Time: 0:35:15 Time:  0:35:15
EUC_particles: 0 hours, 42 mins, 03.90 secs (11:13pm)
Particles removed (final): 1625
ParticleFile_transport: 0 hours, 0 mins, 05.67 secs (11:13pm)
Timer (base): 0 hours, 42 mins, 16.60 secs

_______________________________________________________________________________

Test 7 (2x release longitudes and no DeleteWestward; % increase in time):
_______________________________________________________________________________

Executing: 1979-01-01 00:00:00 to 1979-12-31 00:00:00
Runtime: 364 days
Timestep (dt): 120 minutes
Output (dt): 1 days
Repeat release: 6 days
Depths: 6 dz=50 [50 to 300]
Latitudes: 7 dy=0.8 [-2.4 to 2.4]
Longitudes: 2 [165 190]
Particles (/repeatdt): 84
Particles (total): 5040
Time decorator used.

ofam_fieldset: 0 hours, 0 mins, 01.85 secs (00:21am)
Executing: ParticleFile_1979-1979_v7i
EUC_pset: 0 hours, 6 mins, 34.00 secs (00:28am)
Particles removed: 42
remove_westward_particles: 0 hours, 0 mins, 00.01 secs (00:28am)
INFO: Compiled tparticleAgeAdvectionRK4_3D ==> 
100% (31449600.0 of 31449600.0) |#################| Elapsed Time: 0:43:38 Time:  0:43:38
EUC_particles: 0 hours, 50 mins, 20.67 secs (01:11am)
Particles removed (final): 1841
ParticleFile_transport: 0 hours, 0 mins, 06.70 secs (01:11am)
Timer (base): 0 hours, 50 mins, 29.22 secs

"""