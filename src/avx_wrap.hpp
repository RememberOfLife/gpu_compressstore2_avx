#ifndef AVX_WRAP_CUH
#define AVX_WRAP_CUH

#include <chrono>
#include <cstdint>
#include <immintrin.h>

#ifdef  AVXPOWER

template <typename T>
struct template_type_switch {
    static void process(T* input, uint8_t* mask, T* output, uint64_t N);
};

template <>
struct template_type_switch<uint8_t> {
    static void process(uint8_t* input, uint8_t* mask, uint8_t* output, uint64_t N)
    {
        uint8_t* stop = input+N;
        while (input < stop) {
            // load data and mask
            __m512i a = _mm512_loadu_si512(input);
            __mmask64 k = _load_mask64(reinterpret_cast<__mmask64*>(mask));
            // compressstore into output_p
            _mm512_mask_compressstoreu_epi8(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 64;
            mask += 8;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint16_t> {
    static void process(uint16_t* input, uint8_t* mask, uint16_t* output, uint64_t N)
    {
        uint16_t* stop = input+N;
        while (input < stop) {
            // load data and mask
            __m512i a = _mm512_loadu_si512(input);
            __mmask32 k = _load_mask32(reinterpret_cast<__mmask32*>(mask));
            // compressstore into output_p
            _mm512_mask_compressstoreu_epi16(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 32;
            mask += 4;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint32_t> {
    static void process(uint32_t* input, uint8_t* mask, uint32_t* output, uint64_t N)
    {
        uint32_t* stop = input+N;
        while (input < stop) {
            // load data and mask
            __m512i a = _mm512_loadu_si512(input);
            __mmask16 k = _load_mask16(reinterpret_cast<__mmask16*>(mask));
            // compressstore into output_p
            _mm512_mask_compressstoreu_epi32(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 16;
            mask += 2;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint64_t> {
    static void process(uint64_t* input, uint8_t* mask, uint64_t* output, uint64_t N)
    {
        uint64_t* stop = input+N;
        while (input < stop) {
            // load data and mask
            __m512i a = _mm512_loadu_si512(input);
            __mmask8 k = _load_mask8(reinterpret_cast<__mmask8*>(mask));
            // compressstore into output_p
            _mm512_mask_compressstoreu_epi64(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 8;
            mask += 1;
            output += _mm_popcnt_u64(k);
        }
    }
};

// reverse bits in a byte
uint8_t reverse_byte(uint8_t b) {
   b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
   b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
   b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
   return b;
}

// hostside avx compressstore wrapper for datatypes
template <typename T>
float launch_avx_compressstore(T* input, uint8_t* mask, T* output, uint64_t N) {
    // create temporary mask buffer with reverse bit order per byte (avx req)
    uint8_t* reverse_mask = (uint8_t*)malloc(sizeof(uint8_t) * N/8);
    for (int i = 0; i < N/8; i++) {
        reverse_mask[i] = reverse_byte(mask[i]);
    }
    std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
    template_type_switch<T>::process(input, reverse_mask, output, N);
    std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
    free(reverse_mask);
    return static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
}

#endif

#endif
