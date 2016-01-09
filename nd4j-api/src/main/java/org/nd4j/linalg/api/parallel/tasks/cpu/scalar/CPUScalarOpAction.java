package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.parallel.tasks.Task;
import org.nd4j.linalg.api.parallel.tasks.TaskExecutorProvider;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

public class CPUScalarOpAction extends BaseCPUScalarOpAction {

    /**
     * Constructor for operating on subset of NDArray
     */
    public CPUScalarOpAction(ScalarOp op, int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ) {
        super(op, threshold, n, offsetX, offsetZ, incrX, incrZ);
        if(n % threshold == 0) maxSplits = n / threshold;
        else maxSplits = n / threshold + 1;
        latch = new CountDownLatch(maxSplits);
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public CPUScalarOpAction(ScalarOp op, int threshold) {
        super(op, threshold);
        if(n % threshold == 0) maxSplits = n / threshold;
        else maxSplits = n / threshold + 1;
        latch = new CountDownLatch(maxSplits);
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
