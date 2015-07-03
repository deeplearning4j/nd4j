package org.nd4j.paramserver.pistachio.pb.processor;

interface EventProcessor {
     boolean processEvent(byte[] event);
}
