#!/bin/bash
wget https://sparse.tamu.edu/MM/INPRO/msdoor.tar.gz
tar -zxf msdoor.tar.gz
rm msdoor.tar.gz
mv msdoor/msdoor.mtx ./
rm -rf msdoor/
