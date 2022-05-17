#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fast_prng.hpp"

#ifndef OMP_THREAD_COUNT
#define OMP_THREAD_COUNT 64
#endif

inline void create_bitmask_uniform(uint8_t* mask, uint64_t N, float sel)
{
    #pragma omp parallel for
    for (int t = 0; t < OMP_THREAD_COUNT; t++) {
        fast_prng rng(t);
        uint8_t* my_mask = mask + (t*N/8/OMP_THREAD_COUNT);
        uint64_t my_N = N / OMP_THREAD_COUNT;
        for (int i = 0; i < my_N/8; i++) {
            uint8_t acc = 0;
            for (int j = 7; j >= 0; j--) {
                if (rng.rand() < sel*UINT32_MAX) {
                    acc |= (1<<j);
                }
            }
            reinterpret_cast<uint8_t*>(my_mask)[i] = acc;
        }
    }
}

inline std::vector<uint8_t> create_bitmask(float selectivity, size_t cluster_count, size_t total_elements)
{
    std::vector<bool> bitset;
    bitset.resize(total_elements);
    size_t total_set_one = selectivity * total_elements;
    size_t cluster_size = total_set_one / cluster_count;
    size_t slice = bitset.size() / cluster_count;

    // start by setting all to zero
    #pragma omp parallel for
    for (int t = 0; t < OMP_THREAD_COUNT; t++) {
        for (int i = t*bitset.size()/OMP_THREAD_COUNT; i < (t+1)*bitset.size()/OMP_THREAD_COUNT; i++) {
            bitset[i] = 0;
        }
    }

    for (int i = 0; i < cluster_count; i++) {
        for (int k = 0; k < cluster_size; k++) {
            size_t cluster_offset = i * slice;
            bitset[k + cluster_offset] = 1;
        }
    }

    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements / 8);

    for (int i = 0; i < total_elements / 8; i++) {
        final_bitmask_cpu[i] = 0;
    }

    #pragma omp parallel for
    for (int t = 0; t < OMP_THREAD_COUNT; t++) {
        for (int i = t*bitset.size()/OMP_THREAD_COUNT; i < (t+1)*bitset.size()/OMP_THREAD_COUNT; i++) {
            // set bit of uint8
            if (bitset[i]) {
                uint8_t current = final_bitmask_cpu[i / 8];
                int location = i % 8;
                current = 1 << (7 - location);
                uint8_t add_res = final_bitmask_cpu[i / 8];
                add_res = add_res | current;
                final_bitmask_cpu[i / 8] = add_res;
            }
        }
    }

    return final_bitmask_cpu;
}