#ifndef LLMINFERENCE_H
#define LLMINFERENCE_H

#include "common.h"
#include "llama.h"
#include <string>
#include <vector>

class LLMInference {

    // llama.cpp-specific types
    llama_context* _ctx;
    llama_model* _model;
    llama_sampler* _sampler;
    llama_batch _batch;
    llama_token _currToken;
    
    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> _messages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> _formattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> _promptTokens;
    int _prevLen = 0;

    // stores the complete response for the given query
    std::string _response = "";

    public:

    void loadModel(const std::string& modelPath, float minP, float temperature);

    void addChatMessage(const std::string& message, const std::string& role);
    
    void startCompletion(const std::string& query);

    std::string completionLoop();

    void stopCompletion();

    ~LLMInference();
};

#endif