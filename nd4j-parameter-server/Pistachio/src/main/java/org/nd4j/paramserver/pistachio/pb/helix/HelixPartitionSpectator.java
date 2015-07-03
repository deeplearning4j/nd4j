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
import org.apache.helix.manager.zk.ZKHelixAdmin;
import org.apache.helix.model.IdealState;
import org.apache.helix.model.InstanceConfig;
import org.apache.helix.spectator.RoutingTableProvider;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class HelixPartitionSpectator {

    private static Logger logger = LoggerFactory.getLogger(HelixPartitionSpectator.class);

    private HelixManager manager = null;
    private RoutingTableProvider routingTableProvider = null;
    private final Random rand = new Random();
    private static String[] readExclusionList = ConfigurationManager.getConfiguration().getStringArray("Profile.ReadExclusionList");
    private final ConcurrentHashMap<String, String> host2ip = new ConcurrentHashMap<>(); // cache hostname -> ip mapping
    private static HelixPartitionSpectator helixPartitionSpectator;
    private Long totalPartition = -1L;
    private static Object mutex = new Object();
    private String zkAddress;
    private String helixClusterName;

    private HelixPartitionSpectator(String zkAddr, String clusterName, String instanceName) {
        logger.info("init HelixPartitionSpectator with zkAddr @{}, clusterName {}, instanceName {}",
                zkAddr, clusterName, instanceName);
        zkAddress = zkAddr;
        helixClusterName = clusterName;

        manager = HelixManagerFactory.getZKHelixManager(clusterName, instanceName,
                InstanceType.SPECTATOR, zkAddr);
        try {
            manager.connect();
            routingTableProvider = new RoutingTableProvider();
            manager.addExternalViewChangeListener(routingTableProvider);

        } catch (Exception e) {
            logger.error("caught exception when init HelixPartitionSpectator", e);
            if (manager != null) {
                manager.disconnect();
            }
            throw new RuntimeException("init HelixPartitionSpectator failure");
        }
    }

    public static HelixPartitionSpectator getInstance(String zkAddr, String clusterName, String instanceName) {
        if(helixPartitionSpectator == null){
            synchronized(mutex){
                if(helixPartitionSpectator == null){
                    helixPartitionSpectator = new HelixPartitionSpectator(zkAddr,clusterName,instanceName);
                }
            }
        }
        return helixPartitionSpectator;
    }



    public void close() {
        if (manager != null) {
            manager.disconnect();
        }
    }

    public long getTotalPartition(String resource) {
        if (totalPartition == -1) {
            synchronized(totalPartition) {
                if (totalPartition == -1) {
                    ZKHelixAdmin admin = new ZKHelixAdmin(zkAddress);
                    IdealState idealState = admin.getResourceIdealState(helixClusterName, resource);
                    totalPartition = (long)idealState.getNumPartitions();
                }
            }
        }

        return totalPartition;
    }

    /**
     * Get random one instance name for given partition
     * @param resource
     * @param partition
     * @param state (ONLINE/OFFLINE for OnlineOffline model)
     * @return @Nullable
     */
    public @Nullable String getOneInstanceForPartition(String resource, int partition, String state) {

        List<String> instanceNames = getAllInstance(resource, partition, state);
        if(readExclusionList.length > 0) {
            for(String instanceName : instanceNames) {
                if(!Arrays.asList(readExclusionList).contains(instanceName)) {
                    return instanceName;
                }
                logger.debug("filter ip:" + instanceName +" partition:" + partition);
            }
            return null;
        }else{
            return (instanceNames.isEmpty()) ? null : instanceNames.get(rand.nextInt(instanceNames.size()));
        }

    }

    /**
     * Get all instance names for given partition
     * @param resource
     * @param partition
     * @param state (ONLINE/OFFLINE for OnlineOffline model)
     * @return @Nullable
     */
    public List<String> getAllInstance(String resource, int partition, String state) {
        logger.debug("resource name" + resource +" partition " + partition + " state " + state);
        List<InstanceConfig> instances = routingTableProvider.getInstances(resource, String.format("%s_%d", resource, partition), state);

        List<InstanceConfig> instancesSet = routingTableProvider.getInstances(resource, String.format("%s_%d", resource, partition), "SLAVE");

        for (InstanceConfig instance: instancesSet) {
            String hostname = instance.getHostName();
            logger.debug("all slave instance host {}", hostname);
        }

        List<String> instanceNames = new ArrayList<>(instances.size());
        for (InstanceConfig instance: instances) {
            String hostname = instance.getHostName();
            logger.debug("instance host {}", hostname);
            String ip = host2ip.get(hostname);
            if (ip == null) {
                try {
                    InetAddress addr = InetAddress.getByName(hostname);
                    ip = addr.getHostAddress();
                    host2ip.putIfAbsent(hostname, ip);
                } catch (UnknownHostException ex) {
                    logger.error("Cannot resolve IP for hostname returned by Helix: {}", hostname);
                    if (logger.isDebugEnabled()) {
                        logger.debug("Cannot resolve IP for hostname returned by Helix: {}", hostname, ex);
                    }
                }
            }
            if (ip != null) {
                instanceNames.add(ip);
            }
        }
        return instanceNames;
    }
}
