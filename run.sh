#!/bin/sh
# This script is used to run the Python script with the specified arguments.
python3 train.py --seed 1 --task "halfcheetah-random-v2"
python3 train.py --seed 2 --task "halfcheetah-random-v2"
python3 train.py --seed 3 --task "halfcheetah-random-v2"
python3 train.py --seed 1 --task "hopper-medium-replay-v2"
python3 train.py --seed 2 --task "hopper-medium-replay-v2"
python3 train.py --seed 3 --task "hopper-medium-replay-v2"
python3 train.py --seed 1 --task "walker2d-expert-v2"
python3 train.py --seed 2 --task "walker2d-expert-v2"
python3 train.py --seed 3 --task "walker2d-expert-v2"
python3 train.py --seed 1 --task "halfcheetah-medium-expert-v2"
python3 train.py --seed 2 --task "halfcheetah-medium-expert-v2"
python3 train.py --seed 3 --task "halfcheetah-medium-expert-v2"