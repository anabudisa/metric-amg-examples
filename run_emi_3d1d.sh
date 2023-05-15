#!/bin/bash
REDBG='\e[48;2;255;0;02m'
NC='\e[m'
SAVE_TO_PVD=false
for rds in 0.0 0.2 1.0 5.0
do
  for gma in 1e0 1e2 1e4 1e6 1e8 1e10
  do
    echo -e "\n${REDBG}Assemble and solve EMI 3d-1d problem with radius=$rds and gamma=$gma ${NC}\n"
    python3 ./src/emi_3d1d.py -radius $rds -gamma $gma -dump 1 -outdir ./data/emi_3d1d/radius$rds/gamma$gma/
    mkdir -p ./results/emi_3d1d/radius$rds/gamma$gma/
    echo -e "\n Running solver (output saved to ./results/emi_3d1d/radius$rds/gamma$gma/output.txt)..."
    python3 ./src/run_solver_3d1d.py -infile ./src/input_metric.dat -indir ./data/emi_3d1d/radius$rds/gamma$gma/ -outdir ./results/emi_3d1d/radius$rds/gamma$gma/ > ./results/emi_3d1d/radius$rds/gamma$gma/output.txt
    echo -e "\n Solver done."
    if [ $SAVE_TO_PVD = true ]; then
      python3 ./src/emi_3d1d.py -radius $rds -gamma $gma -load_solution ./results/emi_3d1d/radius$rds/gamma$gma/
    fi
  done
done