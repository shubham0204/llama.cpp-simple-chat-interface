#include "LLMInference.h"
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string modelPath = "/home/shubham/CPP_Projects/llama-cpp-inference/models/smollm2-360m-instruct-q8_0.gguf";
    float temperature = 1.0f;
    float minP = 0.05f;
    std::unique_ptr<LLMInference> llmInference = std::make_unique<LLMInference>();
    llmInference->loadModel(modelPath, minP, temperature);
    llmInference->addChatMessage("You are a helpful assistant", "system");
    while (true) {
        std::cout << "Enter query:\n";
        std::string query;
        std::getline(std::cin, query);
        if (query == "exit") {
            break;
        }
        llmInference->startCompletion(query);
        std::string predictedToken;
        while ((predictedToken = llmInference->completionLoop()) != "[EOG]") {
            std::cout << predictedToken;
            fflush(stdout);
        }
        std::cout << '\n';
    }
    return 0;
}