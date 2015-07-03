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

public interface PistachiosHandler {
    public byte[] lookup(byte[] id, long partitionId, boolean callback) throws Exception;
    public boolean store(byte[] id, long partitionId, byte[] value, boolean callback);
    public boolean processBatch(byte[] id, long partitionId, List<byte[]> events);
    public boolean delete(byte[] id, long partitionId);
    public byte[] getNext(long partitionId, long versionId);
    public void jump(byte[] key, long partitionId, long versionId);
}
