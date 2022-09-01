### Transform the ONNX file before sending it to the compiler
### 1. Import model into FINN with ModelWrapper

import onnx 
import torch 
import os
from preProcessing import *

from qonnx.core.modelwrapper import ModelWrapper

ready_model_filename = "finn_QWNet111.onnx"
model_for_sim = ModelWrapper(ready_model_filename)


dir(model_for_sim)



from qonnx.core.datatype import DataType

finnonnx_in_tensor_name = model_for_sim.graph.input[0].name
finnonnx_out_tensor_name = model_for_sim.graph.output[0].name
print("Input tensor name: %s" % finnonnx_in_tensor_name)
print("Output tensor name: %s" % finnonnx_out_tensor_name)
finnonnx_model_in_shape = model_for_sim.get_tensor_shape(finnonnx_in_tensor_name)
finnonnx_model_out_shape = model_for_sim.get_tensor_shape(finnonnx_out_tensor_name)
print("Input tensor shape: %s" % str(finnonnx_model_in_shape))
print("Output tensor shape: %s" % str(finnonnx_model_out_shape))
finnonnx_model_in_dt = model_for_sim.get_tensor_datatype(finnonnx_in_tensor_name)
finnonnx_model_out_dt = model_for_sim.get_tensor_datatype(finnonnx_out_tensor_name)
print("Input tensor datatype: %s" % str(finnonnx_model_in_dt.name))
print("Output tensor datatype: %s" % str(finnonnx_model_out_dt.name))
print("List of node operator types in the graph: ")
print([x.op_type for x in model_for_sim.graph.node])


## 2. Network preparation: Tidy-up transformations
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants

model_for_sim = model_for_sim.transform(InferShapes())
model_for_sim = model_for_sim.transform(FoldConstants())
model_for_sim = model_for_sim.transform(GiveUniqueNodeNames())
model_for_sim = model_for_sim.transform(GiveReadableTensorNames())
model_for_sim = model_for_sim.transform(InferDataTypes())
model_for_sim = model_for_sim.transform(RemoveStaticGraphInputs())

verif_model_filename = "finn_QWNet111-verification.onnx"
model_for_sim.save(verif_model_filename)

from finn.util.visualization import showInNetron
#showInNetron(verif_model_filename)



# 3 Load dataset into the Brevitas Model

#skipped this is for testing the hardware later


# #BUILD IP WITH FINN
#https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil

model_file = "finn_QWNet111.onnx"

estimates_output_dir = "output_estimates_only"

# #Delete previous run results if exist
print("Generation the IP and the estimated resource reports")
if os.path.exists(estimates_output_dir):
    shutil.rmtree(estimates_output_dir)
    print("Previous run results deleted!")


cfg_estimates = build.DataflowBuildConfig(
    output_dir          = estimates_output_dir,
    mvau_wwidth_max     = 80,
    target_fps          = 1000000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xc7z020clg400-1",
    steps               = build_cfg.estimate_only_dataflow_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ]
)
#%%time
build.build_dataflow_cfg(model_file, cfg_estimates)
print("/n ls -l ./output_estimates_only")
cmd = 'ls -l ./output_estimates_only'
os.system(cmd)
print("/n ls -l ./output_estimates_only/report")
cmd = 'ls -l ./output_estimates_only/report'
os.system(cmd)
print("/n cat output_estimates_only/report/estimate_network_performance.json")
cmd = 'cat output_estimates_only/report/estimate_network_performance.json'
os.system(cmd)

print("\n reading estimate_layer_cycles.json ")
print(read_json_dict("output_estimates_only/report/estimate_layer_cycles.json"))
print("\n reading estimate_layer_resources.json")
print(read_json_dict(estimates_output_dir + "/report/estimate_layer_resources.json"))

# print("Generation the STICHED IP, RTL_SIM, and SYNTH")

# model_file = "finn_QWNet111.onnx"

# rtlsim_output_dir = "output_ipstitch_ooc_rtlsim"

# #Delete previous run results if exist
# if os.path.exists(rtlsim_output_dir):
#     shutil.rmtree(rtlsim_output_dir)
#     print("Previous run results deleted!")

# cfg_stitched_ip = build.DataflowBuildConfig(
#     output_dir          = rtlsim_output_dir,
#     mvau_wwidth_max     = 80,
#     target_fps          = 1000000,
#     synth_clk_period_ns = 10.0,
#     fpga_part           = "xc7z020clg400-1",
#     generate_outputs=[
#         build_cfg.DataflowOutputType.STITCHED_IP,
#         build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
#         build_cfg.DataflowOutputType.OOC_SYNTH,
#     ]
# )

# #%%time
# build.build_dataflow_cfg(model_file, cfg_stitched_ip)