#!/bin/bash

for i in {5..60..5}
do
  echo $i ' Max Eplision ' `date`
  python attack_llcm.py --max_epsilon=$i --max_iter=1
  python attack_llcm.py --max_epsilon=$i --max_iter=5
done
