cmake_minimum_required(VERSION 3.10)
project(llama_inference)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

add_executable(
        chat
        src/llm/LLMInference.cpp
        src/llm/main.cpp
)
target_link_libraries(
        chat
        PRIVATE
        common llama ggml
)