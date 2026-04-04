llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-graph-builder.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            models/afmoe.cpp
            models/apertus.cpp
            models/arwkv7.cpp
            models/bert.cpp
            models/bitnet.cpp
            models/chameleon.cpp
            models/cogvlm.cpp
            models/deepseek2.cpp
            models/delta-net-base.cpp
            models/eurobert.cpp
            models/falcon-h1.cpp
            models/gemma3n-iswa.cpp
            models/gemma4-iswa.cpp
            models/granite-hybrid.cpp
            models/grovemoe.cpp
            models/jamba.cpp
            models/kimi-linear.cpp
            models/lfm2.cpp
            models/mamba-base.cpp
            models/mamba.cpp
            models/minicpm3.cpp
            models/modern-bert.cpp
            models/nemotron-h.cpp
            models/neo-bert.cpp
            models/plamo2.cpp
            models/plm.cpp
            models/qwen35.cpp
            models/qwen35moe.cpp
            models/qwen3next.cpp
            models/qwen3vl-moe.cpp
            models/qwen3vl.cpp
            models/rwkv6-base.cpp
            models/rwkv6.cpp
            models/rwkv6qwen2.cpp
            models/rwkv7-base.cpp
            models/rwkv7.cpp
            models/t5-dec.cpp
            models/t5-enc.cpp
            models/wavtokenizer-dec.cpp
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
