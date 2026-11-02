#!/bin/bash

# ========= Pythia experiments =================
# rapid run -c rapid/configs/kdfp_rebuttal.yaml
# rapid run -c rapid/configs/kd_comps_rebuttal.yaml

# ted --seed=3 --save-path=ted-results/ted-pythia-fwe-bs2-seed3 --model-family=pythia --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --seed=33 --save-path=ted-results/ted-pythia-fwe-bs2-seed33 --model-family=pythia --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --seed=333 --save-path=ted-results/ted-pythia-fwe-bs2-seed333 --model-family=pythia --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5

# ========== Llama 3.2 Experiments ==============
# rapid run -c rapid/configs/kdfp_rebuttal_llama.yaml
# rapid run -c rapid/configs/kd_comps_rebuttal_llama.yaml

ted --seed=3 --save-path=ted-results/ted-llama3.2-fwe-bs2-seed3 --model-family=llama3.2 --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --seed=33 --save-path=ted-results/ted-llama3.2-fwe-bs2-seed33 --model-family=llama3.2 --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --seed=333 --save-path=ted-results/ted-llama3.2-fwe-bs2-seed333 --model-family=llama3.2 --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5


