# HTM
The official implementation of HTM -- Fusion Requires Interaction: A Hybrid Mamba-Transformer Architecture for Deep Interactive Fusion of Multi-modal Images (Under Review)

# 1. Create Environment
conda create -n FusionMamba python=3.8

conda activate HTM

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 

pip install causal_conv1d==1.0.0 # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install mamba_ssm==1.0.1 # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

