#!/bin/bash

if [ $1 == "to_easley" ]; then
  scp $PWD/*.py easley:/users/xbarr/deep-rl/assignment-2
  scp $PWD/*.json easley:/users/xbarr/deep-rl/assignment-2
  scp $PWD/*.sbatch easley:/users/xbarr/deep-rl/assignment-2

elif [ $1 == "from_easley" ]; then
  mkdir -p $PWD/data
  # scp easley:/users/xbarr/deep-rl/assignment-2/results/* $PWD/data/
  scp easley:/users/xbarr/deep-rl/assignment-2/logs/* $PWD/data/
else
  echo "Usage: transfer.bash [to|from]"
fi