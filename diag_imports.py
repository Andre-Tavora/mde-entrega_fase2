import time
t = time.time()
def step(nome):
    print("%-22s %.1fs" % (nome, time.time() - t), flush=True)
import numpy; step("numpy")
import pandas; step("pandas")
import torch; step("torch")
import transformers; step("transformers")
import sentence_transformers; step("sentence_transformers")
print("imports OK", flush=True)
