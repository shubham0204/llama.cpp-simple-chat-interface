#include "LLMInference.h"
#include <memory>
#include <iostream>

void initiateChat() {
    std::string modelPath = "/Users/shubhampanchal/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf";
    float temperature = 0.8f;
    float minP = 0.05f;
    std::unique_ptr<LLMInference> llmInference = std::make_unique<LLMInference>();
    std::string chatTemplate =
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + ' ' + message['content'] + '<|im_end|>' + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant ' }}{% endif %}";
    llmInference->loadModel(modelPath.c_str(), minP, temperature, true, 4096, nullptr, 4, true, false);
    llmInference->addChatMessage("You are a helpful assistant", "system");
    while (true) {
        std::cout << "Enter query:\n";
        std::string query;
        std::getline(std::cin, query);
        if (query == "exit") {
            break;
        }
        llmInference->startCompletion(query.c_str());
        std::string predictedToken;
        while ((predictedToken = llmInference->completionLoop()) != "[EOG]") {
            std::cout << predictedToken;
            fflush(stdout);
        }
        std::cout << '\n';
    }
}


int main(int argc, char *argv[]) {
    initiateChat();
    return 0;
}
