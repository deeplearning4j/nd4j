package org.nd4j.paramserver.pistachio.pb;

import org.nd4j.paramserver.pistachio.pb.kafka.KeyValue;

public interface PistachioIterator {

    KeyValue getNext() throws Exception;
    void jump(byte[] key) throws Exception;

}
