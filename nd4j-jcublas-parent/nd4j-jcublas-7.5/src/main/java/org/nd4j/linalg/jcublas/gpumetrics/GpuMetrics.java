package org.nd4j.linalg.jcublas.gpumetrics;

import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUoccupancyB2DSize;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.utils.KernelLauncher;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;

import static  jcuda.driver.JCudaDriver.*;
import static  jcuda.runtime.JCuda.*;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 See:
 http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class GpuMetrics  {
    public GpuMetrics() {
    }

    private static Logger log = LoggerFactory.getLogger(GpuMetrics.class);
    public final static int MAX_THREADS = 256;
    public final static int MAX_BLOCKS = 64;
    private int gridSize,blockSize,sharedMemory;
    private static  CUoccupancyB2DSize DOUBLE = new CUoccupancyB2DSize() {

        @Override
        public long call(int blockSize) {
            return blockSize * Sizeof.DOUBLE;
        }
    };

    private static CUoccupancyB2DSize FLOAT = new CUoccupancyB2DSize() {
        @Override
        public long call(int blockSize) {
            return blockSize * Sizeof.FLOAT;
        }
    };


    /**
     * Given n, max threads
     * @param n the number of elements to process
     * @param maxThreads the max number of threads
     * @param maxBlocks the max number of blocks
     * @return an array with the number of threads as
     * the first entry and number of blocks
     * as the second entry
     */
    public static int[] getThreadsAndBlocks(int n,int maxThreads,int maxBlocks) {
        //get device capability, to avoid block/grid size exceed the upper bound
        cudaDeviceProp prop = new cudaDeviceProp();
        int[] devicePointer = new int[1];
        cudaGetDevice(devicePointer);
        cudaGetDeviceProperties(prop, devicePointer[0]);


        int threads = (n < maxThreads*2) ? PointerUtil.nextPow2((n + 1) / 2) : maxThreads;
        int blocks = (n + (threads * 2 - 1)) / (threads * 2);


        if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
        {
            throw new IllegalStateException("n is too large, please choose a smaller number!\n");
        }

        if (blocks > prop.maxGridSize[0])
        {
            log.warn("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                    blocks, prop.maxGridSize[0], threads * 2, threads);

            blocks /= 2;
            threads *= 2;
        }


        blocks = Math.min(maxBlocks, blocks);
        return new int[] {threads,blocks};
    }


    /**
     * Get the blocks and threads
     * used for a kernel launch
     * @param dataType the data type
     * @param n the number of elements
     * @return the information used
     * for launching a kernel
     */
    public  static GpuMetrics blockAndThreads(String dataType,int n) {
        //<<<numBlocks, threadsPerBlock>>>
        //<<< gridSize, blockSize >>>
        int size = dataType.equals("double") ? Sizeof.DOUBLE : Sizeof.FLOAT;
        int[] threadsAndBlocks = getThreadsAndBlocks(n,MAX_THREADS,MAX_BLOCKS);
        int sharedMemSize =   (threadsAndBlocks[0] <= 32) ? 2 * threadsAndBlocks[0] * size : threadsAndBlocks[0] * size;
        return new GpuMetrics(threadsAndBlocks[0],threadsAndBlocks[1],sharedMemSize);
    }






}
