echo "===== PR BATCH: runtime (ms) ====="
echo -e "soc-orkut: \c" && ../../build/application/PageRank ../dataset/soc-orkut/soc-orkut.mtx --with-header
echo -e "soc-pokec: \c" && ../../build/application/PageRank ../dataset/soc-pokec/soc-pokec-relationships.mtx --with-header
echo -e "web-uk-05: \c" && ../../build/application/PageRank ../dataset/web-uk-2005/web-uk-2005.mtx --with-header
echo -e "web-wp-09: \c" && ../../build/application/PageRank ../dataset/web-wikipedia-2009/web-wikipedia2009.mtx --with-header
echo -e "kron-21: \c" && ../../build/application/PageRank ../dataset/kron_g500-logn21/kron_g500-logn21.mtx --with-header
echo -e "rgg_n24: \c" && ../../build/application/PageRank ../dataset/rgg_n_2_24_s0/rgg_n_2_24_s0.mtx --with-header
echo -e "roadNet-CA: \c" && ../../build/application/PageRank ../dataset/roadNet-CA/roadNet-CA.mtx --with-header
echo -e "roadNet-TX: \c" && ../../build/application/PageRank ../dataset/roadNet-TX/roadNet-TX.mtx --with-header
echo -e "sc-msdoor: \c" && ../../build/application/PageRank ../dataset/sc-msdoor/msdoor.mtx --with-header
echo -e "sc-ldoor: \c" && ../../build/application/PageRank ../dataset/sc-ldoor/sc-ldoor.mtx --with-header
