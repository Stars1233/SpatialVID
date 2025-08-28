mkdir -p ./checkpoints/
cd ./checkpoints/

# raft 
mkdir -p ./raft/
gdown -c https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O raft/raft-things.pth

# depth anything
huggingface-cli download --resume-download depth-anything/Depth-Anything-V2-Large --local-dir Depth-Anything

# unidepth
huggingface-cli download --resume-download lpiccinelli/unidepth-v2-vitl14 --local-dir UniDepth

# sam
huggingface-cli download --resume-download facebook/sam2.1-hiera-large --local-dir SAM2