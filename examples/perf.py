import torch
import time
import os
from examples.util import print_gpu_memory_gb
from transformers import AutoModelForCausalLM
from torch_memory_saver import torch_memory_saver


MODEL_PATH = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B/48d6d0fc4e02fb1269b36940650a1b7233035cbb/"

# Load model
print_gpu_memory_gb("Before loading model ")
w_alc_s = time.perf_counter()
with torch_memory_saver.region(tag="weights"):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda()
w_alc_t = time.perf_counter() - w_alc_s
print(f"Time to allocate weights: {w_alc_t}")
print_gpu_memory_gb("After loading model and before pause ")

print('sleep...')
time.sleep(3)

# Allcaote dummy KV Cache

print_gpu_memory_gb("Before allocating kv cache")
kv_alc_s = time.perf_counter()
with torch_memory_saver.region("kv_cache"):
    kv_cache = torch.full((16, 2, 32, 8192, 8, 128), 1024,  dtype=torch.float16, device="cuda")
kv_alc_t = time.perf_counter() - kv_alc_s
print(f"Time to allocate kv_cache: {kv_alc_t}")

print_gpu_memory_gb("After allocating kv cache")
w_ps_s = time.perf_counter()
torch_memory_saver.pause(tag="weights")
w_ps_t = time.perf_counter() - w_ps_s
print(f"Time to pause weights: {w_ps_t}")

print_gpu_memory_gb("After pausing weights ")
kv_ps_s = time.perf_counter()
torch_memory_saver.pause(tag="kv_cache")
kv_ps_t = time.perf_counter() - kv_ps_s
print(f"Time to pause kv_cache: {kv_ps_t}")
print_gpu_memory_gb("After pausing kv_cache ")


w_r_s = time.perf_counter()
torch_memory_saver.resume("weights")
w_r_t = time.perf_counter() - w_r_s
print(f"Time to resume weights: {w_r_t}")
print_gpu_memory_gb("After resuming weights ")

kv_r_s = time.perf_counter()
torch_memory_saver.resume("kv_cache")
kv_r_t = time.perf_counter() - kv_r_s
print(f"Time to resume kv_cache: {kv_r_t}")
print_gpu_memory_gb("After resuming kv_cache ")


os._exit(0)
