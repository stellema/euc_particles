#!/usr/bin/env python
import os
rc = os.path.normpath(os.path.join(os.path.expanduser("~"),".ecmwfapirc"))
os.environ['ECMWF_API_URL']="https://api.ecmwf.int/v1"
os.environ['ECMWF_API_KEY']="9961a9b7daba979f74dc3236db1828b9"
os.environ['ECMWF_API_EMAIL']="a.stellema@unsw.edu.au"
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer(url="https://api.ecmwf.int/v1",key="9961a9b7daba979f74dc3236db1828b9",email="a.stellema@unsw.edu.au")
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2012-12-02/to/2012-12-31",
    "expver": "1",
    "grid": "1.5/1.5",
    "levtype": "sfc",
    "param": "180.128/181.128",
    "step": "3",
    "stream": "oper",
    "time": "00:00:00/12:00:00",
    "type": "fc",
    'format'    : "netcdf",
    "target": "output2",
})