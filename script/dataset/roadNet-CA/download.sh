#!/bin/bash
wget https://sparse.tamu.edu/MM/SNAP/roadNet-CA.tar.gz
tar -zxf roadNet-CA.tar.gz
rm roadNet-CA.tar.gz
mv roadNet-CA/roadNet-CA.mtx ./
rm -rf roadNet-CA/
