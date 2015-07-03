package org.nd4j.paramserver.pistachio.pb;

public interface Partitioner {
    long getPartition(byte[] key, long totalParition);
}
