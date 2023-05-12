#!/bin/bash
REDBG='\e[48;2;255;0;02m'
NC='\e[m'
for rds in 0.0 0.2 1.0 5.0
do
  for gma in 1e0 1e2 1e4 1e6 1e8 1e10
  do
    echo -e "\n${REDBG}Assemble EMI 3d-1d problem with radius=$rds and gamma=$gma ${NC}\n"
    python3 ./src/emi_3d1d.py -radius $rds -dump 1 -gamma $gma -outdir ./data/emi_3d1d/radius$rds/gamma$gma/
  done
done