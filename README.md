# Simple LLM Inference in C++ with llama.cpp

## Build

```bash
# clone repository
git clone --depth=1 --recurse-submodules https://github.com/shubham0204/llama.cpp-simple-chat-interface
cd llama.cpp-simple-chat-interface

# download model
wget https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf -P models

# build the executable
mkdir build
cd build
cmake ..
make chat
./chat
```