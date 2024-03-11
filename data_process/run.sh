#!/bin/bash


# DeepCAD
# for i in $(seq 0 18)
# do
#     # Call python script with different interval values
#     timeout 500 python process_abc.py --input /data/xuxiangx/project/abc_step --interval $i --deepcad
#     pkill -f '^python' # cleanup after each run
# done


# ABC
# for i in $(seq 0 99)
# do
#     # Call python script with different interval values
#     timeout 1000 python process_abc.py --input /data/xuxiangx/project/abc_step --interval $i
#     pkill -f '^python' # cleanup after each run
# done

# python deduplicate_cad.py --input deepcad_parsed --deepcad --bit 6
# python deduplicate_cad.py --input abc_parsed --bit 4

# python deduplicate_surfedge.py --input abc_data_split_6bit.pkl --bit 6 --edge
