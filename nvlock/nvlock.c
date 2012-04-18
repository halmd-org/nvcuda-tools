/*
 * nvlock - Exclusively lock an unused NVIDIA device
 *
 * Copyright © 2008-2012  Peter Colberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>

// http://gcc.gnu.org/onlinedocs/cpp/Stringification.html
#define xstr(s) str(s)
#define str(s) #s

#define NVIDIA_DEVICE_FILENAME "/dev/nvidia%d"

static CUresult (*_cuCtxCreate)(CUcontext *, unsigned int, CUdevice) = 0;
static CUresult (*_cuCtxPopCurrent)(CUcontext *) = 0;
static CUresult (*_cuCtxPushCurrent)(CUcontext) = 0;
static int fd = -1;

#define LOG_ERROR(fmt, args...) fprintf(stderr, "nvlock: " fmt "\n", ## args)

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    void *handle;
    char fn[16];
    char *env, *endptr;
    long int val;

    // open dynamic library and load real function symbol
    if (!_cuCtxCreate) {
        handle = dlopen("libcuda.so", RTLD_GLOBAL | RTLD_NOW);
        if (!handle) {
            LOG_ERROR("failed to load libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
        // stringify: CUDA >= 3.2 defines part of its API with macros
        _cuCtxCreate = dlsym(handle, xstr(cuCtxCreate));
        if (!_cuCtxCreate) {
            LOG_ERROR("failed to load symbol " xstr(cuCtxCreate));
            return CUDA_ERROR_UNKNOWN;
        }
        if (dlclose(handle)) {
            LOG_ERROR("failed to unload libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // if environment variable CUDA_DEVICE is set, use integer value as allowed device
    env = getenv("CUDA_DEVICE");
    if (NULL != env && '\0' != *env) {
        val = strtol(env, &endptr, 10);
        if ('\0' != *endptr || val < 0) {
            LOG_ERROR("invalid environment variable CUDA_DEVICE: %s", env);
            return CUDA_ERROR_UNKNOWN;
        }
        if (val != dev) {
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // lock NVIDIA device file with non-blocking request
    snprintf(fn, sizeof(fn), NVIDIA_DEVICE_FILENAME, dev);
    if (-1 != fd) {
        close(fd);
        fd = -1;
    }
    if (-1 == (fd = open(fn, O_RDWR))) {
        LOG_ERROR("failed to open CUDA device in read-write mode: %s", fn);
        return CUDA_ERROR_UNKNOWN;
    }
    if (-1 == flock(fd, LOCK_EX | LOCK_NB)) {
        close(fd);
        fd = -1;
        return CUDA_ERROR_UNKNOWN;
    }

    // create CUDA context
    return _cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxPopCurrent(CUcontext *pctx)
{
    void *handle;

    // open dynamic library and load real function symbol
    if (!_cuCtxPopCurrent) {
        handle = dlopen("libcuda.so", RTLD_GLOBAL | RTLD_NOW);
        if (!handle) {
            LOG_ERROR("failed to load libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
        // stringify: CUDA >= 3.2 defines part of its API with macros
        _cuCtxPopCurrent = dlsym(handle, xstr(cuCtxPopCurrent));
        if (!_cuCtxPopCurrent) {
            LOG_ERROR("failed to load symbol " xstr(cuCtxPopCurrent));
            return CUDA_ERROR_UNKNOWN;
        }
        if (dlclose(handle)) {
            LOG_ERROR("failed to unload libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // unlock the device file
    flock(fd, LOCK_UN);

    // pop current context from CUDA context stack
    return _cuCtxPopCurrent(pctx);
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
    void *handle;
    CUresult ret;
    CUdevice dev;
    char fn[16];

    // open dynamic library and load real function symbol
    if (!_cuCtxPushCurrent) {
        handle = dlopen("libcuda.so", RTLD_GLOBAL | RTLD_NOW);
        if (!handle) {
            LOG_ERROR("failed to load libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
        // stringify: CUDA >= 3.2 defines part of its API with macros
        _cuCtxPushCurrent = dlsym(handle, xstr(cuCtxPushCurrent));
        if (!_cuCtxPushCurrent) {
            LOG_ERROR("failed to load symbol " xstr(cuCtxPopCurrent));
            return CUDA_ERROR_UNKNOWN;
        }
        if (dlclose(handle)) {
            LOG_ERROR("failed to unload libcuda.so");
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // lock the device file with blocking request
    flock(fd, LOCK_EX);

    // push floating context onto CUDA context stack
    return _cuCtxPushCurrent(ctx);
}
