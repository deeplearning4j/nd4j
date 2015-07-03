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

import java.util.ArrayList;

import org.nd4j.paramserver.pistachio.pb.PistachiosServer;

@StateModelInfo(initialState = "OFFLINE", states = {
        "MASTER", "SLAVE", "ERROR", "DROPPED"
    })
/*
@StateModelInfo(initialState = "OFFLINE", states = {
        "ONLINE", "SELFBOOTSTRAP", "BOOTSTRAPOTHER", "ERROR", "DROPPED"
    })
*/
public class BootstrapOnlineOfflineStateModel extends StateModel{
    private static Logger logger = LoggerFactory.getLogger(OnlineOfflineStateModel.class);

    private final AtomicReference<BootstrapPartitionHandler> handler = new AtomicReference<BootstrapPartitionHandler>();
    private final PartitionHandlerFactory handlerFactory;
    private final int partitionId;
    private static final ArrayList<Thread> resetThreadList = new ArrayList<Thread>(256);


    public BootstrapOnlineOfflineStateModel(int partitionId, PartitionHandlerFactory handlerFactory) {
        this.partitionId = partitionId;
        this.handlerFactory = handlerFactory;
    }

    @Transition(to = "SLAVE", from = "OFFLINE")
    public void onBecomeSlaveFromOffline(Message message, NotificationContext context) {
        logger.info("becomes SLAVE from OFFLINE for {}", partitionId);
        //handler.compareAndSet(null, (BootstrapPartitionHandler)handlerFactory.createParitionHandler(partitionId));
        if (handler.compareAndSet(null, (BootstrapPartitionHandler)handlerFactory.createParitionHandler(partitionId))) {
            //handler.get().selfBootstraping();
        if (handler.get() != null) {
        logger.info("start serving {}", partitionId);
            handler.get().startServing();
        }
        else {
        logger.info("null handler{}", partitionId);
        }

        }
    }

    @Transition(to = "SLAVE", from = "MASTER")
    public void onBecomeSlaveFromMaster(Message message, NotificationContext context) {
        logger.info("becomes SLAVE from MASTER for {}", partitionId);
    }

    @Transition(to = "MASTER", from = "SLAVE")
    public void onBecomeMasterFromSlave(Message message, NotificationContext context) {
        logger.info("becomes MASTER from SLAVE for {}", partitionId);
        BootstrapPartitionHandler originHandler = handler.get();
        if (originHandler != null) {
            originHandler.selfBootstraping();
        }
    }

    /*
    @Transition(to = "BOOTSTRAPOTHER", from = "ONLINE")
    public void onBecomeBootstrapotherFromOnline(Message message, NotificationContext context) {
        logger.info("becomes BOOTSTRAPOTHER from ONLINE for {}", partitionId);
        BootstrapPartitionHandler originHandler = handler.get();
        if (originHandler != null) {
            originHandler.bootstrapingOthers();
        }
    }

    @Transition(to = "SELFBOOTSTRAP", from = "BOOTSTRAPOTHER")
    public void onBecomeSelfbootstrapFromBootstrapother(Message message, NotificationContext context) {
        logger.info("becomes SELFBOOTSTRAP from BOOTSTRAPOTHER for {}", partitionId);
        BootstrapPartitionHandler originHandler = handler.get();
        if (originHandler != null) {
            originHandler.selfBootstraping();
        }
    }
    */

    private void stop() {
        BootstrapPartitionHandler originHandler = handler.get();
        if (originHandler != null) {
            logger.info("Stopping for partition {}", partitionId);
            originHandler.stopServing();
            handler.compareAndSet(originHandler, null);
            logger.info("Stopping for partition {} done", partitionId);
        }
    }

    @Transition(to = "OFFLINE", from = "SLAVE")
    public void onBecomeOfflineFromSlave(Message message, NotificationContext context) {
        logger.info("becomes OFFLINE from Slave for {}", partitionId);
        stop();
    }


    @Transition(to = "DROPPED", from = "OFFLINE")
    public void onBecomeDroppedFromOffline(Message message, NotificationContext context) {
        logger.info("becomes DROPPED from OFFLINE for {}", partitionId);
        //stop();

        //TODO: avoid calling static func
        PistachiosServer.getInstance().getLocalStorageEngine().close(partitionId);
    }

    @Transition(to = "OFFLINE", from = "ERROR")
    public void onBecomeOfflineFromError(Message message, NotificationContext context) {
        logger.info("becomes OFFLINE from ERROR for {}", partitionId);
    }

    class ResetThread extends Thread {
        BootstrapOnlineOfflineStateModel model;
        ResetThread(int partitionId0, BootstrapOnlineOfflineStateModel model0) {
            super("Reset Thread "+ partitionId0);
            model = model0;
        }
        public void run() {
            model.stop();
        }
    }

    @Override
    public void reset() {
        logger.info("reset called");
        //stop();
        logger.info("new thread {} created", resetThreadList.size());
        Thread t = new ResetThread(partitionId,this);
        resetThreadList.add(t);
        t.start();
    }

    static public void awaitAllResetThreads() {
        logger.info("awaiting reset threads to finish");
        for (int i =0;i<resetThreadList.size();i++) {
            try {
                logger.info("waiting for thread {}", i);
                resetThreadList.get(i).join(1000000000L);
            } catch (Exception e) {
                logger.info("error ",e);
            }
        }
        logger.info("awaiting reset threads to finish, done.");
    }
}
