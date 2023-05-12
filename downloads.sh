#!/bin/bash
mkdir data
cd data || exit

# google tool that also work with google drive
pip3 install gdown
# gdown is installed under .local/bin
export PATH=$PATH:$HOME/.local/bin

# download meshes for neuron and brain
gdown --folder https://drive.google.com/file/d/1lDWbLFgw-Tfe9X6Zrr3AERYInG36Eukj
cd ..

