# File_name : ort_qnn_htp.py

import onnxruntime as ort
import numpy as np
import time
#Qualcomm utility for pre-/postprocessing of input/outputs in model inference
import io_utils

# Step1: Runtime and model initialization
# Set QNN Execution Provider options.
execution_provider_option = {"backend_path": "QnnHtp.dll",
                              "enable_htp_fp16_precision" : "1",
                           "htp_performance_mode": "high_performance"}

# Create ONNX Runtime session.
onnx_model_path = "./mobilenet_v2.onnx"

session = ort.InferenceSession(onnx_model_path,
                              providers=["QNNExecutionProvider"],
                              provider_options=[execution_provider_option])

# Step2: Input/Output handling, Generate raw input
# github repo for below artifact: https://github.com/quic/wos-ai/tree/main/Artifacts
img_path = "https://raw.githubusercontent.com/quic/wos-ai/refs/heads/main/Artifacts/coffee_cup.jpg"
raw_img = io_utils.preprocess(img_path)


# Model input and output names
outputs = session.get_outputs()[0].name
inputs = session.get_inputs()[0].name

# Step3: Model inferencing using preprocessed input.
start_time = time.time()
for i in range(10):
   prediction = session.run([outputs], {inputs: raw_img})
end_time = time.time()
execution_time = ((end_time-start_time) * 1000)/10

# Step4: Output postprocessing
io_utils.postprocess(prediction)
print("Execution Time: ", execution_time, "ms")
