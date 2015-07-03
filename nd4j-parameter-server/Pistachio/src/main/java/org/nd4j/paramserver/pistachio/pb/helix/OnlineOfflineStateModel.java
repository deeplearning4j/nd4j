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

import java.util.concurrent.atomic.AtomicReference;

import org.apache.helix.NotificationContext;
import org.apache.helix.model.Message;
import org.apache.helix.participant.statemachine.StateModel;
import org.apache.helix.participant.statemachine.StateModelInfo;
import org.apache.helix.participant.statemachine.Transition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@StateModelInfo(initialState = "OFFLINE", states = {
        "ONLINE", "ERROR"
    })
public class OnlineOfflineStateModel extends StateModel {

    private static Logger logger = LoggerFactory.getLogger(OnlineOfflineStateModel.class);

    private final AtomicReference<PartitionHandler> handler = new AtomicReference<PartitionHandler>();
    private final PartitionHandlerFactory handlerFactory;
    private final int partitionId;

    public OnlineOfflineStateModel(int partitionId, PartitionHandlerFactory handlerFactory) {
        this.partitionId = partitionId;
        this.handlerFactory = handlerFactory;
    }

    @Transition(to = "ONLINE", from = "OFFLINE")
    public void onBecomeOnlineFromOffline(Message message, NotificationContext context) {
        logger.info("becomes ONLINE from OFFLINE for {}", partitionId);
        if (handler.compareAndSet(null, handlerFactory.createParitionHandler(partitionId))) {
            handler.get().startServing();
        }
    }

    private void stop() {
        PartitionHandler originHandler = handler.get();
        if (originHandler != null) {
            logger.info("Stopping for partition {}", partitionId);
            originHandler.stopServing();
            handler.compareAndSet(originHandler, null);
            logger.info("Stopping for partition {} done", partitionId);
        }
    }

    @Transition(to = "OFFLINE", from = "ONLINE")
    public void onBecomeOfflineFromOnline(Message message, NotificationContext context) {
        logger.info("becomes OFFLINE from ONLINE for {}", partitionId);
        stop();
    }

    @Transition(to = "DROPPED", from = "OFFLINE")
    public void onBecomeDroppedFromOffline(Message message, NotificationContext context) {
        logger.info("becomes DROPPED from OFFLINE for {}", partitionId);
        stop();
    }

    @Transition(to = "OFFLINE", from = "ERROR")
    public void onBecomeOfflineFromError(Message message, NotificationContext context) {
        logger.info("becomes OFFLINE from ERROR for {}", partitionId);
    }

    @Override
    public void reset() {
        logger.info("reset called");
        stop();
    }
}
