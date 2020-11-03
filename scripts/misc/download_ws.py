#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1981-01-01/to/1981-01-31",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "180.128/181.128",
    "step": "3",
    "stream": "oper",
    "time": "00:00:00/12:00:00",
    "type": "fc",
    "target": "output",
})

# #!/usr/bin/env python
# from ecmwfapi import ECMWFDataServer
# server = ECMWFDataServer()
# server.retrieve({
#     "class": "ei",
#     "dataset": "interim",
#     "date": "1981-01-01/to/2012-12-31",
#     "expver": "1",
#     "grid": "0.75/0.75",
#     "levtype": "sfc",
#     "param": "180.128/181.128",
#     "step": "12",
#     "stream": "oper",
#     "time": "12:00:00",
#     "type": "fc",
#     "target": "output",
# })
# #!/usr/bin/env python
# from ecmwfapi import ECMWFDataServer
# server = ECMWFDataServer()
# server.retrieve({
#     "class": "ei",
#     "dataset": "interim",
#     "date": "19810101/19810201/19810301/19810401/19810501/19810601/19810701/19810801/19810901/19811001/19811101/19811201",
#     "expver": "1",
#     "grid": "0.75/0.75",
#     "levtype": "sfc",
#     "param": "180.128/181.128",
#     "step": "3",
#     "stream": "mnth",
#     "time": "12:00:00",
#     "type": "fc",
#     "target": "output",
# })