#pragma once

#include <chrono>
#include <cstdint>

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
    *popc = val_idx;
    return static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
}
