#include <vector>
#include <algorithm>
#include <iostream>
namespace mcts {
    inline std::vector<uint32_t> encode(const std::vector<int> &binary_state){
        int nChunk = 32;
        std::vector<uint32_t> compress;
        int pos = 0;
        uint32_t num = 0;
        for (uint32_t bit: binary_state) {
            num = (num << 1) + bit;
            pos++;
            if (pos == nChunk) {
                compress.push_back(num);
                num = 0;
                pos = 0;
            }
        }
        if (pos > 0) {
            compress.push_back(num);
        }
        return compress;
    }

    inline std::vector<int> decode(const std::vector<uint32_t> &code, int n_state){
        int nChunk = 32;
        int pos = 0;
        std::vector<int> decompress;

        for (uint32_t chunk: code) {
            std::vector<int> sub_state;
            for (int id = 0; id < nChunk && pos < n_state; id++, pos++) {
                uint32_t digit = chunk & (uint32_t) 1;
                chunk = chunk >> 1;
                sub_state.push_back(digit);
            }
            std::reverse(sub_state.begin(), sub_state.end());
            decompress.insert(decompress.end(), sub_state.begin(), sub_state.end());
        }
        return decompress;
    }
}
