echo "===== SSSP BATCH: runtime (ms) ====="
echo -e "soc-orkut: \c" && ../../build/application/SSSP ../dataset/soc-orkut/soc-orkut.mtx --with-header --src=0
echo -e "soc-pokec: \c" && ../../build/application/SSSP ../dataset/soc-pokec/soc-pokec-relationships.mtx --with-header --src=0
echo -e "web-uk-05: \c" && ../../build/application/SSSP ../dataset/web-uk-2005/web-uk-2005.mtx --with-header --src=0
echo -e "web-wp-09: \c" && ../../build/application/SSSP ../dataset/web-wikipedia-2009/web-wikipedia2009.mtx --with-header --src=0
echo -e "kron-21: \c" && ../../build/application/SSSP ../dataset/kron_g500-logn21/kron_g500-logn21.mtx --with-header --src=0
echo -e "rgg_n24: \c" && ../../build/application/SSSP ../dataset/rgg_n_2_24_s0/rgg_n_2_24_s0.mtx --with-header --src=0
echo -e "roadNet-CA: \c" && ../../build/application/SSSP ../dataset/roadNet-CA/roadNet-CA.mtx --with-header --src=0
echo -e "roadNet-TX: \c" && ../../build/application/SSSP ../dataset/roadNet-TX/roadNet-TX.mtx --with-header --src=0
echo -e "sc-msdoor: \c" && ../../build/application/SSSP ../dataset/sc-msdoor/msdoor.mtx --with-header --src=0
echo -e "sc-ldoor: \c" && ../../build/application/SSSP ../dataset/sc-ldoor/sc-ldoor.mtx --with-header --src=0
