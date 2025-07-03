cmake_minimum_required(VERSION 3.10)
project(llama_inference)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)

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

add_executable(
        chat-vision
        src/vlm/VLMInference.cpp
        src/vlm/main.cpp
)
target_link_libraries(
        chat-vision
        PRIVATE
        common llama ggml
)