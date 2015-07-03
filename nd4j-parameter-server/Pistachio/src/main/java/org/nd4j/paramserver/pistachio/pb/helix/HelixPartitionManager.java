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

package org.nd4j.paramserver.pistachio.pb.helix;

import org.apache.helix.HelixManager;
import org.apache.helix.HelixManagerFactory;
import org.apache.helix.InstanceType;
import org.apache.helix.participant.StateMachineEngine;
import org.apache.helix.participant.statemachine.StateModel;
import org.apache.helix.participant.statemachine.StateModelFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HelixPartitionManager<T extends StateModel> {
    private static Logger logger = LoggerFactory.getLogger(HelixPartitionManager.class);

    private HelixManager manager = null;
    private final String zkAddr;
    private final String clusterName;
    private final String instanceName;

    public HelixPartitionManager(String zkAddr, String clusterName, String instanceName) {
        this.zkAddr = zkAddr;
        this.clusterName = clusterName;
        this.instanceName = instanceName;
    }

    public void start(String stateModelName, StateModelFactory<T> stateModelFactory) {
        try {
            manager = HelixManagerFactory.getZKHelixManager(clusterName, instanceName,
                    InstanceType.PARTICIPANT, zkAddr);

            StateMachineEngine stateMachine = manager.getStateMachineEngine();
            stateMachine.registerStateModelFactory(stateModelName, stateModelFactory);

            manager.connect();
        } catch (Exception e) {
            logger.error("failed to connect manager", e);
            throw new RuntimeException("failed to start HelixPartitionManager");
        }
    }

    public void stop() {
        if (manager != null) {
            manager.disconnect();
        }
    }
}
