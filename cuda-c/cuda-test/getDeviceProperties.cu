#include <cuda.h>
#include <stdio.h>

int main(int argc, char** argv) {

      struct cudaDeviceProp p;

      int device;

      cudaGetDevice(&device);

      cudaGetDeviceProperties(&p, device);

      printf("> %s\n"
             "\ttotalGlobalMem: %u B\n"
             "\tsharedMemPerBlock: %u B\n" "\tregsPerBlock: %d\n"
             "\twarpSize: %d threads\n"
             "\tmemPitch: %u B\n"
             "\tmaxThreadsPerBlock: %d\n"
             "\tmaxThreadsDim: (%d, %d, %d)\n"
             "\tmaxGridSize: (%d, %d, %d)\n"
             "\tclockRate: %d kHz\n"
             "\ttotalConstMem: %u B\n"
             "\tCompute Capability: %d.%d\n"
             "\ttextureAlignment: %u\n"
             "\tdeviceOverlap: %d\n"
             "\tmultiProcessorCount: %d\n"
             "\tkernelExecTimeoutEnabled: %d\n"
             "\tintegrated: %d\n"
             "\tcanMapHostMemory: %d\n"
             "\tcomputeMode: %d\n"
             "\tmaxTexture1D: %d\n"
             "\tmaxTexture2D: (%d, %d)\n"
             "\tmaxTexture3D: (%d, %d, %d)\n"
             "\tmaxTexture1DLayered: (%d, %d)\n"
             "\tmaxTexture2DLayered: (%d, %d, %d)\n"
             "\tsurfaceAlignment: %u\n"
             "\tconcurrentKernels: %d\n"
             "\tECCEnabled: %d\n"
             "\tPCI Bus ID: %d:%d.%d\n"
             "\ttccDriver: %d\n"
             "\tasyncEngineCount: %d\n"
             "\tunifiedAddressing: %d\n"
             "\tmemoryClockRate: %d kHz\n"
             "\tmemoryBusWidth: %d bits\n"
             "\tl2CacheSize: %d B\n"
             "\tmaxThreadsPerMultiProcessor: %d\n",
             p.name, p.totalGlobalMem, p.sharedMemPerBlock, p.regsPerBlock,
             p.warpSize, p.memPitch, p.maxThreadsPerBlock, p.maxThreadsDim[0],
             p.maxThreadsDim[1], p.maxThreadsDim[2], p.maxGridSize[0],
             p.maxGridSize[1], p.maxGridSize[2], p.clockRate, p.totalConstMem,
             p.major, p.minor, p.textureAlignment, p.deviceOverlap,
             p.multiProcessorCount, p.kernelExecTimeoutEnabled, p.integrated,
             p.canMapHostMemory, p.computeMode, p.maxTexture1D,
             p.maxTexture2D[0], p.maxTexture2D[1], p.maxTexture3D[0],
             p.maxTexture3D[1], p.maxTexture3D[2], p.maxTexture1DLayered[0],
             p.maxTexture1DLayered[1], p.maxTexture2DLayered[0],
             p.maxTexture2DLayered[1], p.maxTexture2DLayered[2],
             p.surfaceAlignment, p.concurrentKernels, p.ECCEnabled, p.pciBusID,
             p.pciDeviceID, p.pciDomainID, p.tccDriver, p.asyncEngineCount,
             p.unifiedAddressing, p.memoryClockRate, p.memoryBusWidth,
             p.l2CacheSize, p.maxThreadsPerMultiProcessor);
}
