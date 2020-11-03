# -*- coding: utf-8 -*-
"""
Potential influence of near-surface currents on the global dispersal of marine microplastic
-------------------------------------------------------------------------
David Wichmann, Philippe Delandmeter, Erik van Sebille
d.wichmann@uu.nl
##########################################################################
Objects for data analysis, used for creating figures
"""

import numpy as np
from netCDF4 import Dataset

#Domain of interest
minlon=120.
maxlon=350.
minlat=-15.
maxlat=15


def regions(ddeg, so_lat=-60, na_lat=65):
    """
    Function to return the different region definitions as array
    - ddeg: binning (square, in degree)
    - so_lat: latitutde separating the Southern Ocean from the southern basins
    - na_lat: latitude separating North Atlantic and Arctic regions
    """
    
    Lons = np.arange(0.,360.,ddeg)
    Lats = np.arange(-90.,90.,ddeg)
    
    basinregion={'North': [0,360,0,na_lat],
                 'South': [0,360,0,so_lat]}

    region = np.zeros((len(Lons),len(Lats)))

    for i in range(len(Lats)):
        for j in range(len(Lons)):

            la=Lats[i]
            lo=Lons[j]
            
            (NPminlon, NPmaxlon, NPminlat, NPmaxlat)=basinregion['North']
            if (lo<NPmaxlon and lo>=NPminlon and la<NPmaxlat and la>=NPminlat):
                region[j,i]=1
            (NAminlon, NAmaxlon, NAminlat, NAmaxlat)=basinregion['South']
            if (lo<NAmaxlon and lo>=NAminlon and la<NAmaxlat and la>=NAminlat):
                region[j,i]=2



    region_names = ['North Pacific','South Pacific']
    region=region.T
    region=region.ravel()
    return region, region_names


class ParticleData(object):
    """
    Class that containing 2D particle data and functions to analyse it
    """

    def __init__(self, lons, lats, times, depths=None):
        """
        -params lons, lats, times, depths: arrays containing the data
        """
        
        print('---------------------')
        print('Particle data created')
        print('---------------------')
        print('Particles: ', len(lons))
        print('Snapshots: ', len(lons[0]))
        print('---------------------')
        
        self.lons=lons
        self.lats=lats
        self.times=times
        self.depths=depths
        
    def __del__(self):
        print("Particle Data deleted")
        
    def remove_nans(self): #For removing Nans in problematic regions
        print('Removing NaNs...')
        nan_entries = np.argwhere(np.isnan(self.lons))[:,0]
        indices = [i for i in range(len(self.lons)) if i not in nan_entries]
        print('Removed number of NaN values: ', len(self.lons)-len(indices))
        self.lons = self.lons[indices]
        self.lats = self.lats[indices]
        self.times = self.times[indices]
        
        if self.depths is not None:
            self.depths = self.depths[indices]
        print(self.depths)
        print('NaNs are removed')
        print('---------------------')
        
    def get_distribution(self, t, ddeg):
        """
        Calculate the particle distribution at time t. 
        - param t: integer time from loaded particles
        - param ddeg: binning
        """

        lon_edges=np.linspace(minlon,maxlon,int((maxlon-minlon)/ddeg)+1)        
        lat_edges=np.linspace(minlat,maxlat,int((maxlat-minlat)/ddeg)+1)  
        d , _, _ = np.histogram2d(self.lats[:, t], 
                                  self.lons[:, t], [lat_edges, lon_edges])
        return d

    @classmethod
    def from_nc(cls, pdir, fname, tload=None, Ngrids=40):
        """
        Load 2D data from netcdf particle output files. We assume that there are 
        several output files, each for different initial distributions that have to be merged
        :param pdir: directory of files
        :param fname: file name in pdir
        :param tload: array of times for which we load the data (indices, not actual times)
        :Ngrids: number of different output files to be merged (40 in our case)
        """

        print('Loading data from files: ', pdir + fname)
        
        #Load data from first grid-array
#        i = tload[0]
        pfile = pdir + fname + '.nc'     
        data = Dataset(pfile,'r')
        
        times=data.variables['time']
        lons=data.variables['lon']
        lats=data.variables['lat']
        
#        #Load data from other grid-arrays        
#        for i in range(tload[0], tload[-1]+1):
#            times=np.vstack((times, data.variables['time'][:,i]))
#            lons=np.vstack((lons, data.variables['lon'][:,i]))
#            lats=np.vstack((lats, data.variables['lat'][:,i]))
        times/=86400. #Convert to days

        return cls(lons=lons, lats=lats, times=times)
    


    @classmethod
    def from_nc_3d(cls, pdir, fname, tload=None, Ngrids=40):
        """
        Load 2D data from netcdf particle output files. We assume that there are 
        several output files, each for different initial distributions that have to be merged
        :param pdir: directory of files
        :param fname: file name in pdir
        :param tload: array of times for which we load the data (indices, not actual times)
        :Ngrids: number of different output files to be merged (40 in our case)
        """

        print('Loading data from files: ', pdir + fname)
        import xarray as xr
        #Load data from first grid-array
        i = 0

        pfile = pdir + fname + '.nc'     
        data = xr.open_dataset(pfile, decode_times=False)
        
        times=data['time']
        lons=data['lon']
        lats=data['lat']
        depths=data['z']
#        datax = Dataset(pfile,'r')
#        times=datax.variables['time']
        
        #Load data from other grid-arrays        
        for i in range(1,Ngrids):
            print('Load grid no: ', i)
            pfile = pdir + fname + '.nc'  
            data = Dataset(pfile,'r')
            times=np.vstack((times, data.variables['time'][:,tload]))
            lons=np.vstack((lons, data.variables['lon'][:,tload]))
            lats=np.vstack((lats, data.variables['lat'][:,tload]))
            depths=np.vstack((depths, data.variables['z'][:,tload]))
        times/=86400. #Convert to days

        return cls(lons=lons, lats=lats, times=times, depths=depths)
    
    def set_region_labels(self, ddeg, t, so_lat=-60, na_lat=65):
        """
        Function to give particles labels according to the region they started in
        -param ddeg: binning
        -param t: time for labelling
        """
        r, region_names = regions(ddeg, so_lat=so_lat, na_lat=na_lat)
        
        N=360//ddeg
        label=np.array([int(((la-minlat)//ddeg)*N+(lo-minlon)//ddeg) for la,lo in zip(self.lats[:,t],self.lons[:,t])])
        
        region_label = [r[label[i]] for i in range(len(label))]
        return region_label
        
    

class oceanvector(object):
    """
    Class for ocean vectors. Can be 1d or 2d (lon x lat)
    """
    
    def __init__(self,vec,minlon=minlon,maxlon=maxlon, minlat=minlat,maxlat=maxlat,ddeg=2.):
        """
        -val: A value (e.g. an eigenvalue) which is written as a plot title for figures
        """
        
        self.minlon=minlon
        self.maxlon=maxlon
        self.minlat=minlat
        self.maxlat=maxlat
        self.ddeg = ddeg
        
        #Bin Edges
        self.Lons_edges=np.linspace(minlon,maxlon,int((maxlon-minlon)/ddeg)+1)        
        self.Lats_edges=np.linspace(minlat,maxlat,int((maxlat-minlat)/ddeg)+1)
        
        #Bin centers. This is the format of a field as well.
        self.Lons_centered=np.array([(self.Lons_edges[i]+self.Lons_edges[i+1])/2. for i in range(len(self.Lons_edges)-1)])
        self.Lats_centered=np.array([(self.Lats_edges[i]+self.Lats_edges[i+1])/2. for i in range(len(self.Lats_edges)-1)])        
        
        if vec.ndim==1:
            v1d = vec
            v2d = vec.reshape((len(self.Lats_centered),len(self.Lons_centered)))
        else:
            v1d = vec.ravel()
            v2d = vec
        
        self.V1d = v1d
        self.V2d = v2d