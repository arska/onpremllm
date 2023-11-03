# Aarnos onprem.LLM demo

Using https://github.com/amaiya/onprem with a modified frontend to demo different models and RAG

## Installation

- On my Macbook pro M1 with Mac Os 14 (Sonoma), I used Python 3.11 (3.12 did not work), installed from MacPorts (https://www.macports.org/install.php)
- create a virtual environment to install in: python3 -m virtualenv venv; source venv/bin/activate
- Install PyTorch (https://pytorch.org/get-started/locally/): pip3 install torch torchvision torchaudio
- Install llama-cpp-python with Apple Metal support: CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
- Then install onprem: pip install onprem

## Running

- load the python environment: source venv/bin/activate
- start the WebGUI: streamlit run app.py
