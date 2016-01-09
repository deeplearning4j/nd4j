package org.nd4j.linalg.api.parallel.tasks.cpu.misc;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.parallel.tasks.CPUTaskExecutor;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;
import org.nd4j.linalg.api.shape.tensor.TensorCalculatorFactory;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**Assign one NDArray to another (in parallel), where the shapes of the NDArrays need not match
 * When the array shapes do not match, elements ace copied according to their indices according to their c-order traversal
 * indexes: i.e., [0,0,0], [0,0,1], ..., [0,1,0], ..., [size(0)-1, size(1)-1, size(2)-1]
 *
 * @author Alex Black
 */
public class CPUAssignTask implements Task<Void>, Future<Void> {
    protected final AtomicInteger splitCount = new AtomicInteger(0);
    protected final int maxSplits;
    protected final CountDownLatch latch;
    private volatile boolean cancelled = false;

    private final TensorCalculator tCalcX;
    private final TensorCalculator tCalcZ;
    private final DataBuffer x;
    private final DataBuffer z;

    private final int tLengthX;
    private final int tLengthZ;
    private final int incrX;
    private final int incrZ;

    /**
     * @param x Source array
     * @param z Array to assign to
     */
    public CPUAssignTask(INDArray x, INDArray z){
        tCalcX = TensorCalculatorFactory.getTensorCalculator(x,-1);
        tCalcZ = TensorCalculatorFactory.getTensorCalculator(z,-1);
        maxSplits = tCalcZ.getNumTensors();
        latch = new CountDownLatch(maxSplits);

        tLengthX = tCalcX.getTensorLength();
        tLengthZ = tCalcZ.getTensorLength();
        incrX = tCalcX.getElementWiseStrideForTensor();
        incrZ = tCalcZ.getElementWiseStrideForTensor();

        this.x = x.data();
        this.z = z.data();
    }


    @Override
    public Void invokeBlocking() {
        invokeAsync();
        return blockUntilComplete();
    }

    @Override
    public void invokeAsync() {
        CPUTaskExecutor.getInstance().executeAsync(this);
    }

    @Override
    public Void blockUntilComplete() {
        try {
            return this.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Void call() {
        int thisIdx = splitCount.getAndIncrement(); //Index for tensor along last dimension of Z
        while (thisIdx < maxSplits) {

            int elementZStart = thisIdx * tLengthZ;
            int tensorX = elementZStart / tLengthX;
            int valueInTensorX = elementZStart % tLengthX;

            int offsetZ = tCalcZ.getOffsetForTensor(thisIdx);
            int offsetX = tCalcX.getOffsetForTensor(tensorX);

            if(x.allocationMode() == DataBuffer.AllocationMode.HEAP){
                if(x.dataType() == DataBuffer.Type.FLOAT){
                    float[] xf = (float[])x.array();
                    float[] zf = (float[])z.array();

                    for( int i=0; i<tLengthZ; i++, valueInTensorX++ ){
                        if(valueInTensorX >= tLengthX){ //Get next tensor
                            offsetX = tCalcX.getOffsetForTensor(++tensorX);
                            valueInTensorX = 0;
                        }

                        zf[offsetZ + i*incrZ] = xf[offsetX + valueInTensorX * incrX];
                    }

                } else {
                    double[] xd = (double[])x.array();
                    double[] zd = (double[])z.array();
                    for( int i=0; i<tLengthZ; i++, valueInTensorX++ ){
                        if(valueInTensorX >= tLengthX){ //Get next tensor
                            offsetX = tCalcX.getOffsetForTensor(++tensorX);
                            valueInTensorX = 0;
                        }

                        zd[offsetZ + i*incrZ] = xd[offsetX + valueInTensorX * incrX];
                    }
                }

            } else {
                if(x.dataType() == DataBuffer.Type.FLOAT){
                    FloatBuffer fbx = x.asNioFloat();
                    FloatBuffer fbz = z.asNioFloat();

                    for( int i=0; i<tLengthZ; i++, valueInTensorX++ ){
                        if(valueInTensorX >= tLengthX){ //Get next tensor
                            offsetX = tCalcX.getOffsetForTensor(++tensorX);
                            valueInTensorX = 0;
                        }
                        fbz.put(offsetZ+i*incrZ,fbx.get(offsetX+valueInTensorX*incrX));
                    }

                } else {
                    DoubleBuffer dbx = x.asNioDouble();
                    DoubleBuffer dbz = z.asNioDouble();

                    for( int i=0; i<tLengthZ; i++, valueInTensorX++ ){
                        if(valueInTensorX >= tLengthX){ //Get next tensor
                            offsetX = tCalcX.getOffsetForTensor(++tensorX);
                            valueInTensorX = 0;
                        }
                        dbz.put(offsetZ+i*incrZ,dbx.get(offsetX+valueInTensorX*incrX));
                    }

                }
            }

            latch.countDown();
            thisIdx = splitCount.getAndIncrement();
        }

        return null;
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
        if(!isDone() && !cancelled){
            splitCount.addAndGet(maxSplits);
            while(latch.getCount() > 0) latch.countDown();
            cancelled = true;
            return true;
        }
        return false;
    }

    @Override
    public boolean isCancelled() {
        return cancelled;
    }

    @Override
    public boolean isDone() {
        return latch.getCount() <= 0;
    }

    @Override
    public Void get() throws InterruptedException, ExecutionException {
        latch.await();
        return null;
    }

    @Override
    public Void get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
        latch.await(timeout, unit);
        return null;
    }
}
