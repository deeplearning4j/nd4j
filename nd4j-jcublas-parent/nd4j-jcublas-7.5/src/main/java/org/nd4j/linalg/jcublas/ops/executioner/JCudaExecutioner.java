/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.ops.executioner;


import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.KernelParamsWrapper;
import org.nd4j.linalg.jcublas.util.PointerUtil;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * @author Adam Gibson
 */
public class JCudaExecutioner extends DefaultOpExecutioner {
    private JCudaBuffer dummyFloatPointer, dummyDoublePointer;

    public JCudaExecutioner() {
        try {
            SimpleJCublas.init();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer =KernelFunctions.alloc(new double[]{1});
        parallelExecutioner().setParallelEnabled(false);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).currentResultComplex());
            return Nd4j.scalar(execAndReturn(op).currentResult().doubleValue());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).currentResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }


            INDArray retArray = Nd4j.create(retShape);
            List<CudaContext> contexts = new ArrayList<>();
            List<Accumulation> results = new ArrayList<>();
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                Accumulation op3 = (Accumulation) op2;
                results.add(op3);
                CudaContext ctx = invoke(op3, i * op.z().elementWiseStride(),retArray, false);
                contexts.add(ctx);
                //note here that we are only returning objects to the pool here we aren't actually
                //getting rid of the contexts, this just makes synchronization easier.
                //This also prevents blocking with the object pools.
                //ctx.destroy();

            }

            for(CudaContext ctx : contexts)
                ctx.destroy();

            return retArray;
        }


    }

    @Override
    public INDArray execAndReturn(TransformOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }

    @Override
    protected INDArray execAndReturnVector(TransformOp op, int dimension) {
        return super.execAndReturnVector(op, dimension);
    }

    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }

    @Override
    protected Op exec(Op op, int dimension) {
        return super.exec(op, dimension);
    }

    @Override
    public Op exec(Op op) {
        checkOp(op);
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpreted as vectors)
        if(op.x() instanceof IComplexNDArray
                || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            return super.exec(op);

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t,true);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc,0,Nd4j.scalar(0),true);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc,true);
        }
        return op;
    }

    private JCudaBuffer dummyDouble() {
        return dummyDoublePointer;
    }

    private JCudaBuffer dummyFloat() {
        return dummyFloatPointer;
    }

    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op,true);
        return op.z();
    }



    /**
     * Converts the given parameters
     * in to extra arguments to
     * pass to the kernel
     *
     * @param extraArgs the extra arguments
     * @param dataType  the data type
     * @return
     */
    private JCudaBuffer toArgs(Object[] extraArgs, String dataType) {
        if (dataType.equals("double")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyDouble();
            return KernelFunctions.alloc(PointerUtil.toDoubles(extraArgs));
        } else if (dataType.equals("float")) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyFloat();
            return KernelFunctions.alloc(PointerUtil.toFloats(extraArgs));
        }
        throw new IllegalArgumentException("Illegal datatype");
    }


    private CudaContext invoke(Accumulation op,int i,INDArray result,boolean sync)  {
        checkOp(op);
        CudaContext ctx;

        if(!KernelFunctionLoader.getInstance().exists(op.name()) || executionMode() == ExecutionMode.JAVA || op.isPassThrough())
            super.exec(op);


        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());

        if (op.y() != null) {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }

            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    result,i,metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultOp(op, result)) {
                invokeFunction(op, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
                kParams.close();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }

            return ctx;


        } else {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    result,i,metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultOp(op, result)) {
                invokeFunction(op, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }



        }


        return ctx;
    }
    private CudaContext invoke(ScalarOp op,boolean sync) {
        checkOp(op);
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());


        CudaContext ctx = null;
        if(!KernelFunctionLoader.getInstance().exists(op.name())  || executionMode() == ExecutionMode.JAVA)
            super.exec(op);

        if (op.y() != null) {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }

            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op,metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }



        } else {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }


            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    PointerUtil.getPointer(op),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
                kParams.close();
            }

            catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }


        return ctx;

    }




    private CudaContext invoke(TransformOp op,boolean sync) {
        if(!KernelFunctionLoader.getInstance().exists(op.name()) || op.x() instanceof IComplexNDArray || op.isPassThrough()) {
            super.exec(op);
            return null;
        }
        GpuMetrics metrics = GpuMetrics.blockAndThreads(getType(op),op.n());

        CudaContext ctx;
        if (op.y() != null) {
            int xStride = BlasBufferUtil.getBlasStride(op.x());
            if(xStride < 0) {
                op.setX(op.x().dup());
            }

            int yStride = BlasBufferUtil.getBlasStride(op.y());
            if(yStride < 0) {
                op.setY(op.y().dup());
            }

            /**
             * Construct pointer arguments in the following order:
             * n
             * offset,
             * pointer to buffer
             * increment,
             * extraArgs,
             * result
             */

            Object[] kernelParams = new Object[]{
                    op.n(),
                    op.x().offset(),
                    op.y().offset(),
                    op.x(),
                    op.y(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    BlasBufferUtil.getBlasStride(op.y()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),
                    BlasBufferUtil.getBlasStride(op.z())
                    ,metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();

            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }


        } else {
            Object[] kernelParams = new Object[] {
                    op.n(),
                    op.x().offset(),
                    op.x(),
                    BlasBufferUtil.getBlasStride(op.x()),
                    toArgs(op.extraArgs(), getType(op)),
                    op.z(),metrics.getBlockSize()
            };

            try(KernelParamsWrapper kParams = new KernelParamsWrapper(op,sync,kernelParams).setResultArray(op.z())) {
                invokeFunction(op, metrics,kParams.getContext(), kParams.getKernelParameters());
                ctx = kParams.getContext();
                if(sync)
                    kParams.sync();
            } catch(Exception e) {
                throw new RuntimeException("Could not execute kernel", e);
            }
        }


        return ctx;
    }


    private void invokeFunction(Op op,GpuMetrics metrics,CudaContext cudaContext, Object... kernelParams) {
        /**
         * Invoke a cuda kernel by name. This will be wrt the function name.
         * Functions that are accumulations or transforms have names that end with _strided.
         *
         */
        String functionName = op instanceof TransformOp || op instanceof Accumulation ? op.name() + "_strided" : op.name();
        //force blocks and threads to be even
        KernelFunctions.invoke(
                metrics
                ,functionName
                ,getType(op),cudaContext
                ,kernelParams);

    }

    private String getType(Op op) {
        return op.x().data().dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
    }

}


