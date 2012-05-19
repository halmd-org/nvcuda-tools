/*
 * nvlock - Exclusively lock an unused CUDA device
 *
 * Copyright © 2008-2012 Peter Colberg
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

/* enable GNU extension RTLD_NEXT */
#define _GNU_SOURCE

/* always enable assertions */
#ifdef NDEBUG
# undef NDEBUG
# include <assert.h>
# define NDEBUG
#else
# include <assert.h>
#endif

#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

/* http://gcc.gnu.org/onlinedocs/cpp/Stringification.html */
#define xstr(s) str(s)
#define str(s) #s

/* path to nvidia devices */
#ifndef NVLOCK_DEVICE_PATH
# define NVLOCK_DEVICE_PATH "/dev/nvidia%d"
#endif

/* print error message to stderr */
#define LOG_ERROR(fmt, args...) fprintf(stderr, "nvlock: " fmt "\n", ## args)

/* print debug message to stderr */
#ifdef NDEBUG
# define LOG_DEBUG(fmt, args...)
#else
# define LOG_DEBUG(fmt, args...) fprintf(stderr, "nvlock: " fmt "\n", ## args)
#endif

/* file descriptor of lock on nvidia device */
static int fd = -1;
/* usage count of current CUDA context */
static int use = 0;

/* lock CUDA device */
static CUresult lock_device(CUdevice dev)
{
    char fn[256];
    char *env, *endptr;
    long int val;
    int result;

    /* if environment variable CUDA_DEVICE is set, use integer value as allowed device */
    env = getenv("CUDA_DEVICE");
    if (env != NULL && *env != '\0') {
        val = strtol(env, &endptr, 10);
        if (*endptr != '\0' || val < 0) {
            LOG_ERROR("invalid CUDA_DEVICE environment variable: %s", env);
            return CUDA_ERROR_UNKNOWN;
        }
        if (dev != val) {
            return CUDA_ERROR_UNKNOWN;
        }
    }

    /* lock NVIDIA device file with non-blocking request */
    snprintf(fn, sizeof(fn), NVLOCK_DEVICE_PATH, dev);
    assert(fd == -1);
    fd = open(fn, O_RDWR);
    if (fd == -1) {
        LOG_ERROR("failed to open CUDA device in read-write mode: %s", fn);
        return CUDA_ERROR_UNKNOWN;
    }
    result = flock(fd, LOCK_EX | LOCK_NB);
    if (result == -1) {
        close(fd);
        fd = -1;
        return CUDA_ERROR_UNKNOWN;
    }
    LOG_DEBUG("lock device %i", dev);
    return CUDA_SUCCESS;
}

/* unlock CUDA device */
static void unlock_device()
{
    LOG_DEBUG("unlock device");
    assert(fd != -1);
    close(fd);
    fd = -1;
}

#if CUDA_VERSION >= 4000

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx)
{
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult result;
    CUdevice dev;

    LOG_DEBUG("cuCtxSetCurrent(%p)", ctx);
    cuCtxSetCurrent = dlsym(RTLD_NEXT, xstr(cuCtxSetCurrent));
    if (cuCtxSetCurrent == NULL) {
        LOG_ERROR("failed to resolve symbol " xstr(cuCtxSetCurrent));
        return CUDA_ERROR_UNKNOWN;
    }
    result = cuCtxSetCurrent(ctx);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    /* if context is NULL, decrement usage count */
    if (ctx == NULL) {
        assert(use > 0);
        --use;
    }
    /** if usage count is 0, lock or unlock device */
    if (use == 0) {
        if (ctx == NULL) {
            unlock_device();
        }
        else {
            result = cuCtxGetDevice(&dev);
            if (result != CUDA_SUCCESS) {
                cuCtxSetCurrent(0);
                return result;
            }
            result = lock_device(dev);
            if (result != CUDA_SUCCESS) {
                cuCtxSetCurrent(0);
                return result;
            }
        }
    }
    /* if context is not NULL, increment usage count */
    if (ctx != NULL) {
        ++use;
    }
    return CUDA_SUCCESS;
}

#endif /* CUDA_VERSION >= 4000 */

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult result;

    cuCtxCreate = dlsym(RTLD_NEXT, xstr(cuCtxCreate));
    if (cuCtxCreate == NULL) {
        LOG_ERROR("failed to resolve symbol " xstr(cuCtxCreate));
        return CUDA_ERROR_UNKNOWN;
    }
    result = lock_device(dev);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = cuCtxCreate(pctx, flags, dev);
    if (result != CUDA_SUCCESS) {
        unlock_device();
        return result;
    }
    LOG_DEBUG("cuCtxCreate(%p, %u, %i)", *pctx, flags, dev);
    assert(use == 0);
    ++use;
    return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 3020

# undef cuCtxCreate

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult result;

    cuCtxCreate = dlsym(RTLD_NEXT, xstr(cuCtxCreate));
    if (cuCtxCreate == NULL) {
        LOG_ERROR("failed to resolve symbol " xstr(cuCtxCreate));
        return CUDA_ERROR_UNKNOWN;
    }
    result = lock_device(dev);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = cuCtxCreate(pctx, flags, dev);
    if (result != CUDA_SUCCESS) {
        unlock_device();
        return result;
    }
    LOG_DEBUG("cuCtxCreate(%p, %u, %i)", *pctx, flags, dev);
    assert(use == 0);
    ++use;
    return CUDA_SUCCESS;
}

#endif /* CUDA_VERSION >= 3020 */

CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    CUresult (*cuCtxAttach)(CUcontext *, unsigned int);
    CUresult result;

    cuCtxAttach = dlsym(RTLD_NEXT, xstr(cuCtxAttach));
    if (cuCtxAttach == NULL) {
        LOG_ERROR("failed to resolve symbol " xstr(cuCtxAttach));
        return CUDA_ERROR_UNKNOWN;
    }
    result = cuCtxAttach(pctx, flags);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    LOG_DEBUG("cuCtxAttach(%p, %u)", *pctx, flags);
    if (*pctx == NULL) {
        assert(use == 0);
    }
    else {
        ++use;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx)
{
    CUresult (*cuCtxDetach)(CUcontext);
    CUresult result;

    cuCtxDetach = dlsym(RTLD_NEXT, xstr(cuCtxDetach));
    if (cuCtxDetach == NULL) {
        LOG_ERROR("failed to resolve symbol " xstr(cuCtxDetach));
        return CUDA_ERROR_UNKNOWN;
    }
    result = cuCtxDetach(ctx);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    LOG_DEBUG("cuCtxDetach(%p)", ctx);
    assert(use > 0);
    --use;
    if (use == 0) {
        unlock_device();
    }
    return result;
}
