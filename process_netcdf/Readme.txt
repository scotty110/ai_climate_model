For use by Daniel Giles at University College London.
Data produced by Cyril Morcrette at the Met Office (2021).

There is a data directory for each day (20200101 - 20200110).
 - 80 limited-area models (r01 to r80)
 - 24 time stamps (hourly output)
 - AVG and STD data files for the different data types (see stash codes)

For each day there are 15360 files.

In total we have 76800 profiles.

The filenames are structured as follows:

20200101T0000Z_r01_km1p5_RA2T_224x224sampling_hence_2x2_time022_STD_16004.nc
             1   2     3    4     5                   6       7   8     9
1 date/time
2 which of the 80 limited area models this file is for
3 the grid-length of the model used to produce the data km1p5 means dx = 1.5 km
4 the configuration used for the model 
5 size of region that data is "processed" over. 224x224 means (along with dx=1.5 means we are processing over 336km typical of a climate model grid-box). 
6 Since each limited area model is run over a large area, despite doing 224x224 averaging, we can do that 2x2 times.
7 time since the date/time from [1] in steps of 60 minutes. So time000 = 00:00 midnight, and time023 is 23:00
8 processing that has been done 
  AVG=mean, STD=standard deviation 
9 see below:
  # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 4 list_* variables.
  # ------------------------------
  # 2d fields (these are 2d fields that do NOT vary with time).
  #  0  30 Land-sea mask
  #  0  33 Orography
  # ------------------------------
  # 4d fields (i.e. 3d fields that vary with time)
  #  0  10 qv          (on theta levels)
  #  0 408 pressure    (on theta levels)
  # 16   4 temperature (on theta levels)
  # ------------------------------

Each file is then a simple netcdf file (not loads of metadata).


