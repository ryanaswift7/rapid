#!/bin/bash

ted --save-path=ted-results/ted-pythia-fwe-bs2 --model-family=pythia --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --save-path=ted-results/ted-pythia-bt-fwe-bs2 --model-family=pythia-bt --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --save-path=ted-results/ted-pythia-ftt-fwe-bs2 --model-family=pythia-ftt --dataset=fineweb-edu --lr-filters=3.9e-4 --lr-student=3.66e-5
ted --save-path=ted-results/ted-pythia-500m-fwe-bs2 --model-family=pythia --dataset=fineweb-edu-500m --lr-filters=3.9e-4 --lr-student=3.66e-5