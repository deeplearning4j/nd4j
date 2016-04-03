package org.nd4j.nativeblas;


import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;
import org.nd4j.linalg.api.buffer.util.LibUtils;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Native interface for 
 * op execution on cpu
 * @author Adam Gibson
 *
 * the preload="libnd4j" is there because MinGW puts a "lib" in front of the filename for the DLL, but at load time,
 * we are using the default Windows platform naming scheme, which doesn't put "lib" in front, so that's a bit of a hack,
 * but easier than forcing MinGW into not renaming the library name -- @saudet on 3/21/16
 */
@Platform(include="NativeOps.h", preload="libnd4j", link = "nd4j")
public class NativeOps extends Pointer {
    static {
        Loader.load();
    }

    public NativeOps() {
        allocate();
    }
    private native void allocate();


    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execIndexReduceScalarDouble(long[]extraPointers,int opNum,
                                                       long x,
                                                       long xShapeInfo,
                                                       long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execIndexReduceDouble(long[]extraPointers,int opNum,
                                               long x,
                                               long xShapeInfo,
                                               long extraParams,
                                               long result,
                                               long resultShapeInfoBuffer,
                                               long dimension, int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public native void   execBroadcastDouble(long[]extraPointers,int opNum,
                                             long x,
                                             long xShapeInfo,
                                             long y,
                                             long yShapeInfo,
                                             long result,
                                             long resultShapeInfo,
                                             long dimension, int dimensionLength);



    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public native void   execPairwiseTransformDouble(long[]extraPointers,int opNum,
                                                     long dx,
                                                     int xStride,
                                                     long y,
                                                     int yStride,
                                                     long result,
                                                     int resultStride,
                                                     long extraParams, int n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public native void execPairwiseTransformDouble(long[]extraPointers,
                                                   int opNum,
                                                   long dx,
                                                   long xShapeInfo,
                                                   long y,
                                                   long yShapeInfo,
                                                   long result,
                                                   long resultShapeInfo,
                                                   long extraParams,
                                                   long xIndexes,
                                                   long yIndexes,
                                                   long resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    public native void execPairwiseTransformDouble(
            long[]extraPointers,
            int opNum,
            long dx,
            long  xShapeInfo,
            long y,
            long  yShapeInfo,
            long result,
            long  resultShapeInfo,
            long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceDouble(long[]extraPointers,int opNum,
                                          long x,
                                          long xShapeInfo,
                                          long extraParams,
                                          long result,
                                          long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceDouble(long[]extraPointers,int opNum,
                                          long x,
                                          long xShapeInfo,
                                          long extraParams,
                                          long result,
                                          long resultShapeInfo,
                                          long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native  double execReduceScalarDouble(long[]extraPointers,int opNum,
                                                 long x,
                                                 long xShapeInfo,
                                                 long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduce3Double(long[]extraPointers,int opNum,
                                           long x,
                                           long xShapeInfo,
                                           long extraParamsVals,
                                           long y,
                                           long yShapeInfo,
                                           long result,
                                           long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native double   execReduce3ScalarDouble(long[]extraPointers,int opNum,
                                                   long x,
                                                   long xShapeInfo,
                                                   long extraParamsVals,
                                                   long y,
                                                   long yShapeInfo);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execReduce3Double(long[]extraPointers,int opNum,
                                           long x,
                                           long xShapeInfo,
                                           long extraParamsVals,
                                           long y,
                                           long yShapeInfo,
                                           long result,
                                           long resultShapeInfoBuffer,
                                           long dimension,
                                           int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    public native void   execScalarDouble(long[]extraPointers,int opNum,
                                          long x,
                                          int xStride,
                                          long result,
                                          int resultStride,
                                          double scalar,
                                          long extraParams,
                                          int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    public native void execScalarDouble(long[]extraPointers,int opNum,
                                        long x,
                                        long xShapeInfo,
                                        long result,
                                        long resultShapeInfo,
                                        double scalar,
                                        long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param resultIndexes
     */
    public native void execScalarDouble(long[]extraPointers,int opNum,
                                        long x,
                                        long xShapeInfo,
                                        long result,
                                        long resultShapeInfo,
                                        double scalar,
                                        long extraParams,
                                        int n,
                                        long xIndexes,
                                        long resultIndexes);
    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     */
    public native double   execSummaryStatsScalarDouble(long[] extraPointers, int opNum, long x,
                                                        long xShapeInfo,
                                                        long extraParams, boolean biasCorrected);
    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public native void   execSummaryStatsDouble(long[] extraPointers, int opNum,
                                                long x,
                                                long xShapeInfo,
                                                long extraParams,
                                                long result,
                                                long resultShapeInfo, boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execSummaryStatsDouble(long[]extraPointers,int opNum,long x,
                                                long xShapeInfo,
                                                long extraParams,
                                                long result,
                                                long resultShapeInfoBuffer,
                                                long dimension, int dimensionLength,boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public native void   execTransformDouble(long[]extraPointers,int opNum,
                                             long dx,
                                             int xStride,
                                             long result,
                                             int resultStride,
                                             long extraParams, int n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    public native void   execTransformDouble(long[]extraPointers,int opNum,
                                             long dx,
                                             long xShapeInfo,
                                             long result,
                                             long resultShapeInfo,
                                             long extraParams);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    public native void   execTransformDouble(long[]extraPointers,int opNum,
                                             long dx,
                                             long xShapeInfo,
                                             long result,
                                             long resultShapeInfo,
                                             long extraParams,
                                             long xIndexes,
                                             long resultIndexes);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execIndexReduceScalarFloat(long[]extraPointers,
                                                     int opNum,
                                                     long x,
                                                     long xShapeInfo,
                                                     long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execIndexReduceFloat(long[]extraPointers,int opNum,
                                              long x,
                                              long xShapeInfo,
                                              long extraParams,
                                              long result,
                                              long resultShapeInfoBuffer,
                                              long dimension, int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public native void   execBroadcastFloat(long[]extraPointers,int opNum,
                                            long x,
                                            long xShapeInfo,
                                            long y,
                                            long yShapeInfo,
                                            long result,
                                            long resultShapeInfo,
                                            long dimension, int dimensionLength);



    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public native void   execPairwiseTransformFloat(long[]extraPointers,int opNum,
                                                    long dx,
                                                    int xStride,
                                                    long y,
                                                    int yStride,
                                                    long result,
                                                    int resultStride,
                                                    long extraParams, int n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public native void execPairwiseTransformFloat(long[]extraPointers,int opNum,
                                                  long dx,
                                                  long xShapeInfo,
                                                  long y,
                                                  long yShapeInfo,
                                                  long result,
                                                  long resultShapeInfo,
                                                  long extraParams,
                                                  long xIndexes,
                                                  long yIndexes,
                                                  long resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    public native void execPairwiseTransformFloat(long[]extraPointers,int opNum,
                                                  long dx,
                                                  long  xShapeInfo,
                                                  long y,
                                                  long  yShapeInfo,
                                                  long result,
                                                  long  resultShapeInfo,
                                                  long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceFloat(long[]extraPointers,int opNum,
                                         long x,
                                         long xShapeInfo,
                                         long extraParams,
                                         long result,
                                         long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceFloat(long[]extraPointers,int opNum,
                                         long x,
                                         long xShapeInfo,
                                         long extraParams,
                                         long result,
                                         long resultShapeInfo,
                                         long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native float execReduceScalarFloat(long[]extraPointers,int opNum,
                                              long x,
                                              long xShapeInfo,
                                              long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduce3Float(long[]extraPointers,int opNum,
                                          long x,
                                          long xShapeInfo,
                                          long extraParamsVals,
                                          long y,
                                          long yShapeInfo,
                                          long result,
                                          long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native float   execReduce3ScalarFloat(long[]extraPointers,int opNum,
                                                 long x,
                                                 long xShapeInfo,
                                                 long extraParamsVals,
                                                 long y,
                                                 long yShapeInfo);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execReduce3Float(long[]extraPointers,int opNum,
                                          long x,
                                          long xShapeInfo,
                                          long extraParamsVals,
                                          long y,
                                          long yShapeInfo,
                                          long result,
                                          long resultShapeInfoBuffer,
                                          long dimension,
                                          int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    public native void   execScalarFloat(long[]extraPointers,int opNum,
                                         long x,
                                         int xStride,
                                         long result,
                                         int resultStride,
                                         double scalar,
                                         long extraParams,
                                         int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     */
    public native void execScalarFloat(long[]extraPointers,int opNum,
                                       long x,
                                       long xShapeInfo,
                                       long result,
                                       long resultShapeInfo,
                                       float scalar,
                                       long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public native void execScalarFloat(long[]extraPointers,int opNum,
                                       long x,
                                       long xShapeInfo,
                                       long result,
                                       long resultShapeInfo,
                                       double scalar,
                                       long extraParams,
                                       long xIndexes,
                                       long resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execSummaryStatsScalarFloat(long[]extraPointers,int opNum,long x,
                                                      long xShapeInfo,
                                                      long extraParams,boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStatsFloat(long[]extraPointers,int opNum,
                                               long x,
                                               long xShapeInfo,
                                               long extraParams,
                                               long result,
                                               long resultShapeInfo,boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public native void   execSummaryStatsFloat(long[]extraPointers,int opNum,long x,
                                               long xShapeInfo,
                                               long extraParams,
                                               long result,
                                               long resultShapeInfoBuffer,
                                               long dimension, int dimensionLength,boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public native void   execTransformFloat(long[]extraPointers,int opNum,
                                            long dx,
                                            int xStride,
                                            long result,
                                            int resultStride,
                                            long extraParams, int n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    public native void   execTransformFloat(long[]extraPointers,int opNum,
                                            long dx,
                                            long xShapeInfo,
                                            long result,
                                            long resultShapeInfo,
                                            long extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public native void   execTransformFloat(long[]extraPointers,int opNum,
                                            long dx,
                                            long xShapeInfo,
                                            long result,
                                            long resultShapeInfo,
                                            long extraParams,
                                            long xIndexes,
                                            long resultIndexes);


    /**
     * Append an input array
     * to the end of a flat array
     * in a particular order
     * @param offset the offset of the array to start at
     * @param order the order
     * @param result the result array
     * @param resultShapeInfo the shape info for te array
     * @param input the input for the array
     * @param inputShapeInfo the shape information for that array
     */
    public native void flattenFloat(int offset,
                               char order,
                               long result,
                               long resultShapeInfo,
                               long input,
                               long inputShapeInfo);


    /**
     * Append an input array
     * to the end of a flat array
     * in a particular order
     * @param offset the offset of the array to start at
     * @param order the order
     * @param result the result array
     * @param resultShapeInfo the shape info for te array
     * @param input the input for the array
     * @param inputShapeInfo the shape information for that array
     */
    public native void flattenDouble(int offset,
                                    char order,
                                    long result,
                                    long resultShapeInfo,
                                    long input,
                                    long inputShapeInfo);

    /**
     * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
     */
    public native void initializeDevicesAndFunctions();

    public native  long mallocHost(long memorySize, int flags);

    public native  long mallocDevice(long memorySize, long ptrToDeviceId, int flags);

    public native  long freeHost(long pointer);

    public native  long freeDevice(long pointer, long deviceId);
}
