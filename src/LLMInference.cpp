#include "LLMInference.h"
#include <cstring>
#include <iostream>

void LLMInference::loadModel(const std::string& model_path, float min_p, float temperature) {
    // create an instance of llama_model
    llama_model_params modelParams = llama_model_default_params();
    _model = llama_load_model_from_file(model_path.data(), modelParams);

    if (!_model) {
        throw std::runtime_error("load_model() failed");
    }

    // create an instance of llama_context
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 0;               // take context size from the model GGUF file
    ctxParams.no_perf = true;          // disable performance metrics
    _ctx = llama_new_context_with_model(_model, ctxParams);

    if (!_ctx) {
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    // initialize sampler
    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    samplerParams.no_perf = true;      // disable performance metrics
    _sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();
}

void LLMInference::addChatMessage(const std::string& message, const std::string& role) {
    _messages.push_back({strdup(role.data()), strdup(message.data())});
}

void LLMInference::startCompletion(const std::string& query) {
    addChatMessage(query, "user");

    // apply the chat-template 
    int newLen = llama_chat_apply_template(
            _model,
            nullptr,
            _messages.data(),
            _messages.size(),
            true,
            _formattedMessages.data(),
            _formattedMessages.size()
    );
    if (newLen > (int)_formattedMessages.size()) {
        // resize the output buffer `_formattedMessages`
        // and re-apply the chat template
        _formattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(_model, nullptr, _messages.data(), _messages.size(), true, _formattedMessages.data(), _formattedMessages.size());
    }
    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::start_completion() failed");
    }
    std::string prompt(_formattedMessages.begin() + _prevLen, _formattedMessages.begin() + newLen);
    _promptTokens = common_tokenize(_model, prompt, true, true);

    // create a llama_batch containing a single sequence
    // see llama_batch_init for more details
    _batch.token = _promptTokens.data();
    _batch.n_tokens = _promptTokens.size();
}

std::string LLMInference::completionLoop() {
    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    int contextSize = llama_n_ctx(_ctx);
    int nCtxUsed = llama_get_kv_cache_used_cells(_ctx);
    if (nCtxUsed + _batch.n_tokens > contextSize) {
        std::cerr << "context size exceeded" << '\n';
        exit(0);
    }
    // run the model
    if (llama_decode(_ctx, _batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    _currToken = llama_sampler_sample(_sampler, _ctx, -1);
    if (llama_token_is_eog(_model, _currToken)) {
        addChatMessage(strdup(_response.data()), "assistant");
        _response.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    _response += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached
    // in the KV cache
    _batch.token = &_currToken;
    _batch.n_tokens = 1;

    return piece;
}


void LLMInference::stopCompletion() {
    _prevLen = llama_chat_apply_template(
            _model,
            nullptr,
            _messages.data(),
            _messages.size(),
            false,
            nullptr,
            0
    );
    if (_prevLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::stop_completion() failed");
    }
}


LLMInference::~LLMInference() {
    // free memory held by the message text in messages
    // (as we had used strdup() to create a malloc'ed copy)
    for (llama_chat_message &message: _messages) {
        delete message.content;
    }
    llama_kv_cache_clear(_ctx);
    llama_sampler_free(_sampler);
    llama_free(_ctx);
    llama_free_model(_model);
}