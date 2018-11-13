#!/bin/bash
wget http://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
gunzip soc-pokec-relationships.txt.gz
sed -i '1i1632803 1632803 30622564' soc-pokec-relationships.txt
mv soc-pokec-relationships.txt soc-pokec-relationships.mtx
