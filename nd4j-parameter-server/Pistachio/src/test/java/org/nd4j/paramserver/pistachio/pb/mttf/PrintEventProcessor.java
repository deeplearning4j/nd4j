package org.nd4j.paramserver.pistachio.pb.mttf;

import org.nd4j.paramserver.pistachio.pb.EventProcessor;

import java.util.List;

/**
 * @author Adam Gibson
 */
public class PrintEventProcessor implements EventProcessor {
    @Override
    public void processBatch(byte[] id, long partitionId, List<byte[]> events) {
        System.out.println(new String(id));
    }
}
