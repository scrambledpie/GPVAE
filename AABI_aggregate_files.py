import subprocess
import sys
import os
import shutil

files = os.listdir("/home/maths/phrnaj/GPVAE_checkpoints/")
files = [f for f in files if "NP" in f or "SIN" in f]

for f in files:
    number = f.split(":")[0]

    if "NP" in f:
        elbo = "NP"
    elif "SIN" in f:
        elbo = "SIN"
    
    for b in ["1.0", "20.0", "50.0", "100.0"]:
        if b in f:
            beta = b


    new_file = elbo + "_" + b + "_" + number
    new_file = "/home/maths/phrnaj/AABIsavefiles/" + new_file
    
    src_file = "/home/maths/phrnaj/GPVAE_checkpoints/" + f + "/res/ELBO_pandas"
    
    new_file_ = shutil.copy(src_file, new_file)
    
    