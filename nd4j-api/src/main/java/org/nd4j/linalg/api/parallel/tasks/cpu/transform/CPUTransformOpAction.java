package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.concurrent.CountDownLatch;

public class CPUTransformOpAction extends BaseCPUTransformOpAction {

    public CPUTransformOpAction(TransformOp op, int threshold){
        super(op,threshold);

        if(n % threshold == 0) maxSplits = n / threshold;
        else maxSplits = n / threshold + 1;
        latch = new CountDownLatch(maxSplits);

        if (incrX == -1) {
            //Edge case: sometimes NDArray.elementWiseStride() returns -1, due to weird strides,
            //but every element is still separated by same amount in buffer
            //For example, a TransformOp with x.length() == x.data.length(), but x.stride() is not ascending/descending
            INDArray reshapeX = op.x().reshape(new int[]{1, ArrayUtil.prod(op.x().shape())});
            incrX = reshapeX.stride(1);
        }
        if (incrY == -1) {
            if (op.y() == op.x()) incrY = incrX;
            else {
                INDArray reshapeY = op.y().reshape(new int[]{1, ArrayUtil.prod(op.y().shape())});
                incrY = reshapeY.stride(1);
            }
        }
        if (incrZ == -1) {
            if (op.z() == op.x()) incrZ = incrX;
            else {
                INDArray reshapeZ = op.z().reshape(new int[]{1, ArrayUtil.prod(op.z().shape())});
                incrY = reshapeZ.stride(1);
            }
        }
    }
    

    @Override
    public Void call() {

        int thisIdx = splitCount.getAndIncrement();
        while(thisIdx < maxSplits ) {
            final int startIdx = thisIdx * threshold;
            final int endIdx = (thisIdx == maxSplits - 1 ? n : startIdx + threshold);  //End index, exclusive

            doOp(startIdx,endIdx,offsetX,offsetY,offsetZ);

            //For task completion:
            latch.countDown();

            thisIdx = splitCount.getAndIncrement();
        }

        return null;
    }

}
