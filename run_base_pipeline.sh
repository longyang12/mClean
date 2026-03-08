# tmux new -s process
# tmux ls
# tmux a -t process

# CPU
# export HF_HUB_OFFLINE=0
# export TRANSFORMERS_OFFLINE=0


# GPU
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

#######################################################
#########             nocache                 #########
#######################################################
# echo =========arXiv start at nocache========
# # arXiv
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-arxiv-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-arxiv-refine-cpu-np1-nocache.yaml


# sleep 120
# echo =========stackexchange start at nocache========


# # stackexchange
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-stackexchange-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-stackexchange-refine-cpu-np1-nocache.yaml


# sleep 120
# echo =========wiki start at nocache========


# # wiki
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-wiki-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-wiki-refine-cpu-np1-nocache.yaml


# sleep 120
# echo =========github start at nocache========


# # github
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-code-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-code-refine-cpu-np1-nocache.yaml


# sleep 120
# echo =========c4 start at nocache========


# # c4
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-c4-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-c4-refine-cpu-np1-nocache.yaml



# sleep 120
# echo =========llava-pretrain cpu start at nocache========


# # llava-pretrain cpu
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/llava-pretrain-refine-cpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/llava-pretrain-refine-cpu-np1-nocache.yaml


# sleep 120
# echo =========llava-pretrain gpu start at nocache========


# # llava-pretrain gpu
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/llava-pretrain-refine-gpu-np1-nocache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/llava-pretrain-refine-gpu-np1-nocache.yaml



# sleep 120
# echo =========msr-vtt cpu start at nocache========


# msr-vtt cpu
# first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-cpu-np1-nocache.yaml
# sleep 60

# second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-cpu-np1-nocache.yaml

# sleep 120
# echo =========msr-vtt gpu start at nocache========


# msr-vtt gpu
# first clean
python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-gpu-np1-nocache.yaml
sleep 60

# second clean
python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-gpu-np1-nocache.yaml


#######################################################
##########             cache                 ##########
#######################################################
# sleep 120
# echo =========arXiv start at cache========
# # arXiv
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-arxiv-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-arxiv-refine-cpu-np1-cache.yaml


# sleep 120
# echo =========stackexchange start at cache========


# # stackexchange
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-stackexchange-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-stackexchange-refine-cpu-np1-cache.yaml


# sleep 120
# echo =========wiki start at cache========


# # wiki
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-wiki-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-wiki-refine-cpu-np1-cache.yaml


# sleep 120
# echo =========github start at cache========


# # github
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-code-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-code-refine-cpu-np1-cache.yaml


# sleep 120
# echo =========c4 start at cache========


# # c4
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-c4-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/redpajama-c4-refine-cpu-np1-cache.yaml


# sleep 120
# echo =========msr-vtt cpu start at cache========


# # msr-vtt cpu
# # first clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-cpu-np1-cache.yaml
# sleep 60

# # second clean
# python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-cpu-np1-cache.yaml



sleep 120
echo =========msr-vtt gpu start at cache========


# msr-vtt gpu
# first clean
python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-gpu-np1-cache.yaml
sleep 60

# second clean
python tools/process_data.py --config /inspire/ssd/project/smartsoftware/yanglong-253108120134/project/data-juicer/data-recipes/msr-vtt-refine-gpu-np1-cache.yaml


