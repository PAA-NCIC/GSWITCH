#!/bin/bash
wget https://sparse.tamu.edu/MM/DIMACS10/rgg_n_2_24_s0.tar.gz
tar -zxf rgg_n_2_24_s0.tar.gz
mv rgg_n_2_24_s0/rgg_n_2_24_s0.mtx ./
rm rgg_n_2_24_s0.tar.gz
rm -rf rgg_n_2_24_s0/

