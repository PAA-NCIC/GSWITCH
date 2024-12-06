cuda_add_executable(BFS bfs.cu)
target_link_libraries(BFS gswitch)

cuda_add_executable(MBFS mbfs.cu)
target_link_libraries(MBFS gswitch)

cuda_add_executable(CC cc.cu)
target_link_libraries(CC gswitch)

cuda_add_executable(PageRank pagerank.cu)
target_link_libraries(PageRank gswitch)

cuda_add_executable(BC bc.cu)
target_link_libraries(BC gswitch)

cuda_add_executable(SSSP sssp.cu)
target_link_libraries(SSSP gswitch)

cuda_add_executable(GC gc.cu)
target_link_libraries(GC gswitch)

#cuda_add_executable(BF bf.cu)
#target_link_libraries(BF gswitch)

#cuda_add_executable(Delta-PageRank delta-pagerank.cu)
#target_link_libraries(Delta-PageRank gswitch)
#
#cuda_add_executable(Test test.cu)
#target_link_libraries(Test gswitch)
