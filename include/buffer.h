#ifndef __BENCHMARK_BUFFER_H__
#define __BENCHMARK_BUFFER_H__


// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <memory>
#include <cstddef>
#include <cstdint>
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_dma.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <new>
#include <cstdlib>
#include <iostream>
#include "util.h"

using error = std::runtime_error;
using std::string;



typedef std::shared_ptr<nvm_dma_t> DmaPtr;

typedef std::shared_ptr<void> BufferPtr;



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);


BufferPtr createBuffer(size_t size);


BufferPtr createBuffer(size_t size, int cudaDevice);

static void getDeviceMemory(int device, void*& bufferPtr, void*& devicePtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    size += 64*1024;
    //std::cout << "DMA Size: "<< size << std::endl;

//	printf("getDeviceMemroy size = %ld\n", size);
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }
    /*
    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    */

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }
//	printf("getDeviceMemroy origPtr = %p\n", origPtr);

    origPtr = bufferPtr;
    //devicePtr = (void*) (((uint64_t)attrs.devicePointer));
    devicePtr = (void*) ((((uint64_t)attrs.devicePointer) + (64*1024)) & 0xffffffffff0000);
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (64*1024))  & 0xffffffffff0000);
	
//	printf("getDeviceMemroy bufferPtr = %p\n", bufferPtr);
}

static void getDeviceMemory2(int device, void*& bufferPtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    //devicePtr = nullptr;
    size += 128;
//	printf("getDeviceMemroy2 size = %ld\n", size);
	
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

//	printf("getDeviceMemroy2 malloc = %p\n", &bufferPtr);
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

//	printf("getDeviceMemroy2 memset = %p\n", bufferPtr);
	
    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
/*
    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    devicePtr = (void*) (((uint64_t)attrs.devicePointer));
*/
//	printf("getDeviceMemroy2 origPtr = %p\n", bufferPtr);

    origPtr = bufferPtr;
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (128))  & 0xffffffffffffe0);

//	printf("getDeviceMemroy2 bufferPtr = %p\n", bufferPtr);

    //std::cout << "getdeviceMemory: " << std::hex << bufferPtr <<  std::endl;
}

static void getDeviceMemory(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory(device, bufferPtr, notUsed, size, notUsed);
}

/*
static void getDeviceMemory2(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory2(device, bufferPtr, size);
}
*/




inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;
    void* buffer = nullptr;

    /*
    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }
    */

    int err  = posix_memalign(&buffer, 4096, size);

    if (err) {
        throw error(string("Failed to allocate host memory: ") + std::to_string(err));
    }
    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    if (!nvm_ok(status))
    {
        //cudaFreeHost(buffer);
        free(buffer);
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, [buffer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        //cudaFreeHost(buffer);
        free(buffer);
    });
}



inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    //std::cout << "Got Device mem\n";
    int status = nvm_dma_map_device(&dma, ctrl, bufferPtr, size);
    //std::cout << "Got dma_map_devce status\n";
	//printf("status = %d\n", status);
	
    if (!nvm_ok(status))
    {
        //std::cout << "Got dma_map_devce failed\n";
        //cudaFree(bufferPtr);
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
	//printf("memset = %d\n", err);

    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

	//printf("vaddr = %p\n", dma->vaddr);

    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(origPtr);
        //std::cout << "Deleting DMA\n";
    });
}



inline BufferPtr createBuffer(size_t size)
{
    void* buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    return BufferPtr(buffer, [](void* ptr) {

        cudaFreeHost(ptr);
    });
}



inline BufferPtr createBuffer(size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createBuffer(size);
    }

    void* bufferPtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory2(cudaDevice, bufferPtr, size, origPtr);
    //std::cout << "createbuffer: " << std::hex << bufferPtr <<  std::endl;

    return BufferPtr(bufferPtr, [origPtr](void* ptr) {
        __ignore(ptr);
        cudaFree(origPtr);
        //std::cout << "Deleting Buffer\n";
    });
}

/*

DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    return createDma(ctrl, size);
}



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    return createDma(ctrl, size, cudaDevice);
}

*/
#endif
