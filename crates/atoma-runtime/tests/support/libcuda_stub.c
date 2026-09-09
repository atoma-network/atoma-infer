/*
 * A stub `libcuda.so`: the eight driver entry points a RuntimeContext open and one free-memory
 * reading make, and nothing else.
 *
 * cudarc resolves the driver by name through dlopen under the workspace's fallback-dynamic-loading
 * pin, so a shared object called libcuda.so ahead of the real one on the loader's search path
 * answers those calls. Every device here is imaginary and every count is a constant, which is the
 * point: a reading's value says which device's context the call was made against.
 *
 * Built and loaded by tests/free_memory.rs; see that file for how it reaches the loader.
 */

#include <stddef.h>
#include <stdint.h>

typedef int CUdevice;
typedef void *CUcontext;
typedef int CUresult;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_INVALID_CONTEXT 201

/* Two devices, so a reading taken while the other device's context is current is distinguishable. */
#define DEVICE_COUNT 2
static const size_t FREE_BYTES[DEVICE_COUNT] = {1000, 2000};
static const size_t TOTAL_BYTES = 999999;

/* A context handle: non-null, distinct per device, and recognisable in a debugger. */
static CUcontext context_of(CUdevice device) {
    return (CUcontext)(uintptr_t)(4096 * (device + 1));
}

/* The driver's current context is per-thread, and this crate's thread discipline depends on it. */
static __thread CUcontext current = NULL;

CUresult cuInit(unsigned int flags) {
    (void)flags;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (ordinal < 0 || ordinal >= DEVICE_COUNT) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

/* Every attribute reads zero; the only one asked for is memory-pool support, which stays off. */
CUresult cuDeviceGetAttribute(int *value, int attribute, CUdevice device) {
    (void)attribute;
    if (device < 0 || device >= DEVICE_COUNT) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *value = 0;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
    if (device < 0 || device >= DEVICE_COUNT) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *context = context_of(device);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    if (device < 0 || device >= DEVICE_COUNT) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *context) {
    *context = current;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext context) {
    current = context;
    return CUDA_SUCCESS;
}

/* The reading is the current context's device, which is what makes an unbound read observable. */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    for (CUdevice device = 0; device < DEVICE_COUNT; device++) {
        if (context_of(device) == current) {
            *free = FREE_BYTES[device];
            *total = TOTAL_BYTES;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_CONTEXT;
}
