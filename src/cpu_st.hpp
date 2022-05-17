#pragma once

#include <chrono>
#include <cstdint>
#include <bitset>

template <typename T>
float launch_cpu_single_thread(T* input, uint8_t* mask, T* output, uint64_t N, uint64_t* popc)
{
    std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
    uint64_t val_idx = 0;
    for (int i = 0; i < N/8; i++) {
        uint32_t acc = reinterpret_cast<uint8_t*>(mask)[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i*8 + (7-j);
            bool v = 0b1 & (acc>>j);
            if (v) {
                output[val_idx++] = input[idx];
            }
        }
    }
    std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
    if (popc) {
        *popc = val_idx;
    }
    return static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
}

inline uint64_t buf_popc(uint8_t* mask, uint64_t N)
{
    uint64_t* themask = (uint64_t*)mask;
    uint64_t lpopc = 0;
    for (uint64_t i = 0; i < N/64; i++) {
        lpopc += std::bitset<64>(themask[i]).count();
    }
    
    for (uint64_t i = (N/64) * 64; i < N; i++) {
        uint64_t bi = i / 8;
        uint64_t bo = i % 8;
        if ((mask[bi] >> (7 - bo)) & 0b1) {
            lpopc++;
        }
    }
    return lpopc;
}