#!/bin/bash
wget https://sparse.tamu.edu/MM/DIMACS10/kron_g500-logn21.tar.gz
tar -zxf kron_g500-logn21.tar.gz
mv kron_g500-logn21/kron_g500-logn21.mtx ./
rm kron_g500-logn21.tar.gz
rm -rf kron_g500-logn21/
