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

import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;

public class DefaultDataInterpreter implements PistachioDataInterpreter {
    static PistachioDataInterpreter dataInterpreter = null;

    public static final String COMMA = ",";

    public static PistachioDataInterpreter getDataInterpreter() {
        if (dataInterpreter == null) {
            if (ConfigurationManager.getConfiguration().getString("data.interpreter") !=  null) {
                try {
                    Class<?> clazz = Class.forName(ConfigurationManager.getConfiguration().getString("data.interpreter"));
                    dataInterpreter = (PistachioDataInterpreter)clazz.newInstance();
                } catch (Exception e) {
                }
            }
            if (dataInterpreter == null) {
                dataInterpreter = new DefaultDataInterpreter();
            }
        }
        return dataInterpreter;
    }

    @Override
    public String interpretId(byte[] id) {
        if (id.length == 8)
            return new Long(com.google.common.primitives.Longs.fromByteArray(id)).toString();

        return java.util.Arrays.toString(id);
    }

    @Override
    public String interpretData(byte[] data) {
        return java.util.Arrays.toString(data);
    }

    @Override
    public String interpretIds(List<byte[]> ids) {
        StringBuilder sb = new StringBuilder();
        for (byte[] id: ids) {
            sb.append(interpretId(id)).append(COMMA);
        }
        return sb.toString();
    }
}
