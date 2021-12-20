import os, sys
from dataset import DatasetHCPRest, DatasetYADRest, prepare_HCPRest_timeseries, prepare_YADRest_timeseries

if __name__=='__main__':
    #ds = DatasetHCPRest()
    #prepare_YADRest_timeseries()
    yad_ds = DatasetYADRest()
    print("Done")