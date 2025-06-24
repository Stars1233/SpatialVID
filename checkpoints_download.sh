mkdir -p ./checkpoints/
cd ./checkpoints/

export HF_ENDPOINT=https://hf-mirror.com
# depth anything
huggingface-cli download --resume-download depth-anything/Depth-Anything-V2-Large --local-dir Depth-Anything

# raft 
mkdir -p ./raft/
gdown -c https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft/raft-things.pth

# unidepth
huggingface-cli download --resume-download lpiccinelli/unidepth-v2-vitl14 --local-dir UniDepth
