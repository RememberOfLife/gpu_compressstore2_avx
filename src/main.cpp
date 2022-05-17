#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>


#include <assert.h>
#include <omp.h>

#include "avx_wrap.hpp"
#include "cpu_st.hpp"
#include "mask_gen.hpp"

#define OMP_THREAD_LIMIT 64

#define VALIDATION true

#ifndef DATASUBSET
#define DATASUBSET false
#endif

#define REPS 5

FILE* output;

enum MASK_TYPE {
    MASK_TYPE_UNIFORM,
    MASK_TYPE_CLUSTER,
    MASK_TYPE_MULTICLUSTER,
};

const char* mask_str[3] = {
    "uniform",
    "cluster",
    "multi-cluster",
};

template <typename T> struct bufs {
    uint64_t N;
    T* in;
    T* out1;
    T* out2;
    uint8_t* mask;
    bufs(uint64_t N) : N(N)
    {
        in = (T*)malloc(N * sizeof(T));
        out1 = (T*)malloc(N * sizeof(T));
        out2 = (T*)malloc(N * sizeof(T));
        mask = (uint8_t*)malloc(N / 8);
        // put some data into the input
        create_bitmask_uniform((uint8_t*)in, N*sizeof(T)*8, 0.5);
    }
    ~bufs()
    {
        free(in);
        free(out1);
        free(out2);
        free(mask);
    }
};

template <typename T>
void benchmark(uint64_t N, MASK_TYPE mt, float ms, const char* type_str)
{
    bufs<T> b(N);
    // gen mask and data
    switch (mt) {
        case MASK_TYPE_UNIFORM: {
            create_bitmask_uniform(b.mask, b.N, ms);
        } break;
        case MASK_TYPE_CLUSTER: {
            std::vector<uint8_t> gmask = create_bitmask(ms, 1, N);
            memcpy(b.mask, &gmask[0], b.N / 8);
        } break;
        case MASK_TYPE_MULTICLUSTER: {
            std::vector<uint8_t> gmask = create_bitmask(ms, 4, N);
            memcpy(b.mask, &gmask[0], b.N / 8);
        } break;
        default: {
            fprintf(stderr, "UNKNOWN MASK TYPE");
            exit(1);
        } break;
    }

    uint64_t popc;

    // run cpu singlethreaded
    for (int i = 0; i < REPS; i++) {
        float t_cpu_st = launch_cpu_single_thread(b.in, b.mask, b.out1, b.N, &popc);
        fprintf(
            output, "cpu_st;%s;%lu;%s;%f;%f;\n", type_str, b.N, mask_str[mt], ms,
            t_cpu_st);
    }

    // run avx, if enabled
#ifdef AVXPOWER
    for (int i = 0; i < REPS; i++) {
        float t_cpu_avx = launch_avx_compressstore(b.in, b.mask, b.out2, N);
        fprintf(
            output, "cpu_avx;%s;%lu;%s;%f;%f;\n", type_str, b.N, mask_str[mt], ms,
            t_cpu_avx);
    }
    if (VALIDATION && memcmp(b.out1, b.out2, popc * sizeof(T)) != 0) {
        fprintf(stderr, "VALIDATION FAILURE\n");
        exit(1);
    }
#endif
}

template <typename T>
void mt_benchmark(uint64_t N, MASK_TYPE mt, float ms, const char* type_str)
{
    bufs<T> b(N);
    // gen mask and data
    switch (mt) {
        case MASK_TYPE_UNIFORM: {
            create_bitmask_uniform(b.mask, b.N, ms);
        } break;
        case MASK_TYPE_CLUSTER: {
            std::vector<uint8_t> gmask = create_bitmask(ms, 1, N);
            memcpy(b.mask, &gmask[0], b.N / 8);
        } break;
        case MASK_TYPE_MULTICLUSTER: {
            std::vector<uint8_t> gmask = create_bitmask(ms, 4, N);
            memcpy(b.mask, &gmask[0], b.N / 8);
        } break;
        default: {
            fprintf(stderr, "UNKNOWN MASK TYPE");
            exit(1);
        } break;
    }

    uint64_t popc;

    launch_cpu_single_thread(b.in, b.mask, b.out2, b.N, &popc);
    // run cpu multihtreaded
    for (int i = 0; i < REPS; i++) {
        std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
        const unsigned int TC = omp_get_max_threads();//std::thread::hardware_concurrency();
        uint64_t lpopc[OMP_THREAD_LIMIT];
        uint64_t elems_per_thread = b.N / TC;
        elems_per_thread = (elems_per_thread / 8) * 8;
        uint64_t overhang_elems = b.N - (TC-1)*elems_per_thread;
        // use omp parallel for, otherwise we spawn threads and everything gets slower than singlethreaded
        #pragma omp parallel for
        for (int i = 0; i < TC; i++) {
            lpopc[i] = buf_popc(b.mask + i*elems_per_thread/8, i == TC-1 ? overhang_elems : elems_per_thread);
        }
        for (int i = 0; i < TC-1; i++) {
            lpopc[i + 1] += lpopc[i];
        }
        #pragma omp parallel for
        for (int i = 0; i < TC; i++) {
            launch_cpu_single_thread(b.in + i*elems_per_thread, b.mask + i*elems_per_thread/8, b.out1 + (i == 0 ? 0 : lpopc[i - 1]), i == TC-1 ? overhang_elems : elems_per_thread, NULL);
        }
        std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
        float t_cpu_mt = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
        fprintf(
            output, "cpu_mt;%s;%lu;%s;%f;%f;\n", type_str, b.N, mask_str[mt], ms,
            t_cpu_mt);
    }
    if (VALIDATION && memcmp(b.out1, b.out2, popc * sizeof(T)) != 0) {
        fprintf(stderr, "MT VALIDATION FAILURE\n");
        assert(0);
        exit(1);
    }

    // run avx, if enabled
#ifdef AVXPOWER
    for (int i = 0; i < REPS; i++) {
        std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
        const unsigned int TC = omp_get_max_threads();//std::thread::hardware_concurrency();
        uint64_t lpopc[OMP_THREAD_LIMIT];
        uint64_t elems_per_thread = b.N / TC;
        elems_per_thread = (elems_per_thread / 8) * 8;
        uint64_t overhang_elems = b.N - (TC-1)*elems_per_thread;
        // use omp parallel for, otherwise we spawn threads and everything gets slower than singlethreaded
        #pragma omp parallel for
        for (int i = 0; i < TC; i++) {
            lpopc[i] = buf_popc(b.mask + i*elems_per_thread/8, i == TC-1 ? overhang_elems : elems_per_thread);
        }
        for (int i = 0; i < TC-1; i++) {
            lpopc[i + 1] += lpopc[i];
        }
        #pragma omp parallel for
        for (int i = 0; i < TC; i++) {
            launch_avx_compressstore(b.in + i*elems_per_thread, b.mask + i*elems_per_thread/8, b.out1 + (i == 0 ? 0 : lpopc[i - 1]), i == TC-1 ? overhang_elems : elems_per_thread);
        }
        std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
        float t_cpu_avx = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
        fprintf(
            output, "cpu_m_avx;%s;%lu;%s;%f;%f;\n", type_str, b.N, mask_str[mt], ms,
            t_cpu_avx);
    }
    if (VALIDATION && memcmp(b.out1, b.out2, popc * sizeof(T)) != 0) {
        fprintf(stderr, "VALIDATION FAILURE\n");
        exit(1);
    }
#endif
}

template <typename T> void benchmark_type(const char* type_str)
{
    printf("type: %s", type_str);
    for (uint64_t N = DATASUBSET ? 1<<30 : 1<<10; N <= (1 << 30); N *= 4) {
        float ms = 0.5;
#if !DATASUBSET
        for (ms = 0.1; ms < 1.0; ms += 0.1) {
            benchmark<T>(N, MASK_TYPE_CLUSTER, ms, type_str);
            printf(".");
            benchmark<T>(N, MASK_TYPE_MULTICLUSTER, ms, type_str);
            printf(".");
#endif
            benchmark<T>(N, MASK_TYPE_UNIFORM, ms, type_str);
            printf(".");
            mt_benchmark<T>(N, MASK_TYPE_UNIFORM, ms, type_str);
            printf(".");
#if !DATASUBSET
            mt_benchmark<T>(N, MASK_TYPE_CLUSTER, ms, type_str);
            printf(".");
            mt_benchmark<T>(N, MASK_TYPE_MULTICLUSTER, ms, type_str);
            printf(".");
        }
#endif
    }
    printf("\n");
}

int main()
{
    output = fopen("./cpu_data.csv", "w+");
    fprintf(
        output,
        "approach;data type;element count;mask distribution kind;selectivity;runtime "
        "(ms);\n");
    if (!output) {
        printf("could not open file 1\n");
        exit(1);
    }

    setbuf(stdout, NULL);
    
    omp_set_num_threads(omp_get_max_threads());

    // both uint8_t and uint16_t need AVX512_VBMI2 which we don't have ;'(
    benchmark_type<uint32_t>("uint32_t");
    benchmark_type<uint64_t>("uint64_t");

    fclose(output);
    return 0;
}
