#include <cstdio>
#include <cstring>

#include "avx_wrap.hpp"
#include "cpu_st.hpp"
#include "mask_gen.hpp"

#define VALIDATION true

#define REPS 10

FILE* cpu_st;
#ifdef AVXPOWER
FILE* cpu_avx;
#endif

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

    // run cpu singlethreaded
    float t_cpu_st = 0;
    t_cpu_st += launch_cpu_single_thread(b.in, b.mask, b.out1, b.N);
    t_cpu_st /= REPS;
    fprintf(
        cpu_st, "%s;%lu;%s;%f;%f;\n", type_str, b.N, mask_str[mt], ms,
        t_cpu_st);

    // run avx, if enabled
#ifdef AVXPOWER
    float t_cpu_avx = 0;
    t_cpu_avx += launch_avx_compressstore(b.in, b.mask, b.out2, N);
    t_cpu_avx /= REPS;
    if (VALIDATION && memcmp(b.out1, b.out2, N * sizeof(T)) != 0) {
        fprintf(stderr, "VALIDATION FAILURE\n");
        exit(1);
    }
    fprintf(
        cpu_avx, "|%s|%lu|%s|%f|%f|\n", type_str, b.N, mask_str[mt], ms,
        t_cpu_avx);
#endif
}

template <typename T> void benchmark_type(const char* type_str)
{
    printf("type: %s", type_str);
    for (uint64_t N = 1024; N <= (1 << 20); N *= 2) {
        for (float ms = 0.1; ms < 1.0; ms += 0.1) {
            benchmark<T>(N, MASK_TYPE_UNIFORM, ms, type_str);
            benchmark<T>(N, MASK_TYPE_CLUSTER, ms, type_str);
            benchmark<T>(N, MASK_TYPE_MULTICLUSTER, ms, type_str);
            printf(".");
        }
    }
    printf("\n");
}

int main()
{
    cpu_st = fopen("./cpu_st.csv", "w+");
    if (!cpu_st) {
        printf("could not open file 1\n");
        exit(1);
    }
#ifdef AVXPOWER
    cpu_st = fopen("./data/cpu_avx.csv", "w+");
    if (!cpu_st) {
        printf("could not open file 2\n");
        exit(1);
    }
#endif

    setbuf(stdout, NULL);

    benchmark_type<uint8_t>("uint8_t");
    benchmark_type<uint16_t>("uint16_t");
    benchmark_type<uint32_t>("uint32_t");
    benchmark_type<uint64_t>("uint64_t");

    fclose(cpu_st);
#ifdef AVXPOWER
    fclose(cpu_avx);
#endif
    return 0;
}
