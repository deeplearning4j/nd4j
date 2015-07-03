/*
 * Copyright 2014 Yahoo! Inc. Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
 * agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.nd4j.paramserver.pistachio.pb.kafka;

import kafka.utils.VerifiableProperties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.ByteBufferOutput;


public class KeyValueEncoder implements kafka.serializer.Encoder<KeyValue> {
    private static Logger logger = LoggerFactory.getLogger(KeyValueEncoder.class);
    private static final ThreadLocal<ByteBufferOutput> threadByteBuffer =
        new ThreadLocal<ByteBufferOutput>() {
            @Override protected ByteBufferOutput initialValue() {
                return new ByteBufferOutput(10240);
            }
        };
    Kryo kryo = new Kryo();

    public KeyValueEncoder(VerifiableProperties props) {
    }

    public byte[] toBytes(KeyValue keyValue) {
        try {

            threadByteBuffer.get().clear();

            kryo.writeObject(threadByteBuffer.get(), keyValue);
            //output.close();



            return threadByteBuffer.get().toBytes();
        } catch (Throwable t) {
            logger.error("Exception in encode:", t);
            return "".getBytes();
        }
    }
}
