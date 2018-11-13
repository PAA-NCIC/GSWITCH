#!/bin/bash
wget https://sparse.tamu.edu/MM/SNAP/roadNet-TX.tar.gz
tar -zxf roadNet-TX.tar.gz
rm roadNet-TX.tar.gz
mv roadNet-TX/roadNet-TX.mtx ./
rm -rf roadNet-TX/

