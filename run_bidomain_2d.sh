#!/bin/bash
REDBG='\e[48;2;255;0;02m'
NC='\e[m'
#echo -e "${REDBG}Running bidomain_2d.py${NC}"
prcnd="hazmath hazmath_Schwarz hazmath_HEM metric_mono"
for pr in $prcnd
do
  for gma in 1e0 1e2 1e4 1e6 1e8 1e10
  do
    echo -e "\n${REDBG}Running bidomain_2d.py with $pr preconditioner and gamma=$gma ${NC}\n"
    python3 ./src/bidomain_2d.py -nrefs 5 -precond "$pr" -gamma $gma
  done
done

