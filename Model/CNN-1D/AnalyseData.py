import os
from preProcessing import *

#Analyse the data out of the compiler

estimates_output_dir = "output_estimates_only"
print("ls -l ./output_estimates_only")
cmd = 'ls -l ./output_estimates_only'
os.system(cmd)
print("ls -l ./output_estimates_only/report")
cmd = 'ls -l ./output_estimates_only/report'
os.system(cmd)
print("cat output_estimates_only/report/estimate_network_performance.json")
cmd = 'cat output_estimates_only/report/estimate_network_performance.json'
os.system(cmd)
print("reading estimate_layer_cycles.json ")
print(read_json_dict("output_estimates_only/report/estimate_layer_cycles.json"))
print("reading estimate_layer_resources.json")
print(read_json_dict(estimates_output_dir + "/report/estimate_layer_resources.json"))