cd app;
conda create -n py38 python=3.8;
conda activate py38;
sudo apt install nvidia-cuda-toolkit;
pip install torch;
pip install transformers;
pip install streamlit;
