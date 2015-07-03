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


package org.nd4j.paramserver.pistachio.pb;
import java.util.List;
import java.util.Map;

import java.util.concurrent.Future;
import org.nd4j.paramserver.pistachio.pb.exception.*;

public interface PistachiosClientImpl {
    byte[] lookup(byte[] id, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    boolean store(byte[] id, byte[] value, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    boolean processBatch(byte[] id, List<byte[]> events) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    void close();

    Map<byte[], byte[]> multiLookup(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception;

    Future<Map<byte[], byte[]>> multiLookupAsync(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    Future<Map<byte[], Boolean>> multiProcessAsync(Map<byte[], byte[]> events) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    Future<Boolean> storeAsync(byte[] id, byte[] value) throws MasterNotFoundException, ConnectionBrokenException, Exception;
    boolean delete(byte[] id) throws MasterNotFoundException, ConnectionBrokenException;
    PistachioIterator iterator(long partitionId);
}
