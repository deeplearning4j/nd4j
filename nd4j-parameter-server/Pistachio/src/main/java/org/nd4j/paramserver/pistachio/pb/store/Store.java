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

package org.nd4j.paramserver.pistachio.pb.store;

//import org.zeromq.ZMQ;

public interface Store {
    public boolean add(byte[] msg, long offset);
    public void update(byte[] msg, long offset);
    public byte[] get(byte[] msg);
    public boolean delete(byte[] msg, long offset);
    public long getCurrentOffset();
    public boolean open(int partition);
    public boolean close();
    public boolean commitOffset(long offset);
    //public void transfer(ZMQ.Socket sender);
    public void flush();
}
