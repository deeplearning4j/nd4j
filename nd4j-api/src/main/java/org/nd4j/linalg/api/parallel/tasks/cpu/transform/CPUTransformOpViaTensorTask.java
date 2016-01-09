package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;
import org.nd4j.linalg.api.shape.tensor.TensorCalculatorFactory;

import java.util.concurrent.CountDownLatch;

public class CPUTransformOpViaTensorTask extends BaseCPUTransformOpAction {
    private TensorCalculator tCalcx;
    private TensorCalculator tCalcy;
    private TensorCalculator tCalcz;
    private final int maxSplits;
    private final int tensorLength;
    private final int blocksPerTensor;
    private final int blockSize;


    public CPUTransformOpViaTensorTask(TransformOp op, int threshold){
        super(op,threshold);

        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        int tensorDim;
        if(y==null){
            if(x==z){
                //x=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
            } else {
                //z=Op(x)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, z);
            }
        } else {
            if(x==z){
                //x=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y);
            } else {
                //z=Op(x,y)
                tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, y, z);
            }
        }

        tCalcx = TensorCalculatorFactory.getTensorCalculator(x,tensorDim);
        if(y != null) tCalcy = TensorCalculatorFactory.getTensorCalculator(y,tensorDim);
        if(x == z) tCalcz = tCalcx;
        else tCalcz = TensorCalculatorFactory.getTensorCalculator(z,tensorDim);

        tensorLength = tCalcx.getTensorLength();
        if(tensorLength <= threshold) blocksPerTensor = 1;
        else blocksPerTensor = tensorLength / threshold;

        blockSize = tensorLength / blocksPerTensor; //Except for last, might be a bit bigger due to rounding

        maxSplits = tCalcx.getNumTensors() * blocksPerTensor;
        latch = new CountDownLatch(maxSplits);

        this.incrX = tCalcx.getElementWiseStrideForTensor();
        this.incrY = (tCalcy == null ? 0 : tCalcy.getElementWiseStrideForTensor());
        if(x == z) incrZ = incrX;
        else incrZ = tCalcz.getElementWiseStrideForTensor();
    }

    @Override
    public Void call() {
        INDArray x = op.x();
        INDArray z = op.z();

        int thisIdx = splitCount.getAndIncrement();
        while(thisIdx < maxSplits){
            int tensorIdx = thisIdx / blocksPerTensor;
            int blockWithinTensor = thisIdx % blocksPerTensor;

            int offsetX = tCalcx.getOffsetForTensor(tensorIdx);
            int offsetY = (tCalcy != null ? tCalcy.getOffsetForTensor(tensorIdx) : 0);
            int offsetZ = ( x == z ? offsetX : tCalcz.getOffsetForTensor(tensorIdx));

            int startIdx = blockWithinTensor * blockSize;
            int endIdx;
            if(blockWithinTensor == blocksPerTensor -1 ) endIdx = tensorLength;
            else endIdx = startIdx + blockSize;

            doOp(startIdx,endIdx,offsetX,offsetY,offsetZ);


            latch.countDown();
            thisIdx = splitCount.getAndIncrement();
        }

        return null;
    }
}
