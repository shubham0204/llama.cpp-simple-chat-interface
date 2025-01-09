cmake_minimum_required(VERSION 3.10)
project(llama_inference)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/llama.cpp")

add_executable(
    chat
    src/LLMInference.cpp src/main.cpp
)
target_link_libraries(
    chat 
    PRIVATE
    common llama ggml
)