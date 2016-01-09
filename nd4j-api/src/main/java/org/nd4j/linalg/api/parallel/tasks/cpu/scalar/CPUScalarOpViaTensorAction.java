package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskFactory;
import org.nd4j.linalg.api.parallel.tasks.TaskFactoryProvider;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;
import org.nd4j.linalg.api.shape.tensor.TensorCalculatorFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

public class CPUScalarOpViaTensorAction extends BaseCPUScalarOpAction {
    private TensorCalculator tCalcx;
    private TensorCalculator tCalcz;
    private final int tensorLength;
    private final int blocksPerTensor;
    private final int blockSize;

    public CPUScalarOpViaTensorAction(ScalarOp op, int threshold){
        super(op,threshold);

        INDArray x = op.x();
        INDArray z = op.z();

        int tensorDim;
        if(x==z){
            //x=Op(x,y)
            tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x);
        } else {
            //z=Op(x,y)
            tensorDim = OpExecutionerUtil.chooseElementWiseTensorDimension(x, z);
        }

        tCalcx = TensorCalculatorFactory.getTensorCalculator(x, tensorDim);
        if(x == z) tCalcz = tCalcx;
        else tCalcz = TensorCalculatorFactory.getTensorCalculator(z,tensorDim);

        tensorLength = tCalcx.getTensorLength();
        if(tensorLength <= threshold) blocksPerTensor = 1;
        else blocksPerTensor = tensorLength / threshold;

        blockSize = tensorLength / blocksPerTensor; //Except for last, might be a bit bigger due to rounding

        maxSplits = tCalcx.getNumTensors() * blocksPerTensor;
        latch = new CountDownLatch(maxSplits);

        this.incrX = tCalcx.getElementWiseStrideForTensor();
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
            int offsetZ = ( x == z ? offsetX : tCalcz.getOffsetForTensor(tensorIdx));

            int startIdx = blockWithinTensor * blockSize;
            int endIdx;
            if(blockWithinTensor == blocksPerTensor -1 ) endIdx = tensorLength;
            else endIdx = startIdx + blockSize;

            doOp(startIdx,endIdx,offsetX,0,offsetZ);


            latch.countDown();
            thisIdx = splitCount.getAndIncrement();
        }

        return null;
    }
}
