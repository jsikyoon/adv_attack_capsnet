#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED="True"

TAG='log'

LOG="${TAG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./train.py