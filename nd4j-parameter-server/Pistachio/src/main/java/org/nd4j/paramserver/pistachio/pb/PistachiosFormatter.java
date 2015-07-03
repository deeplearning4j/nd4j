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


import com.google.common.base.Joiner;
import kafka.admin.CreateTopicCommand;
import kafka.utils.ZKStringSerializer$;
import kafka.utils.ZkUtils;
import org.I0Itec.zkclient.ZkClient;
import org.apache.commons.cli.*;
import org.apache.helix.HelixDefinedState;
import org.apache.helix.HelixException;
import org.apache.helix.manager.zk.ZKHelixAdmin;
import org.apache.helix.model.ClusterConstraints.ConstraintAttribute;
import org.apache.helix.model.ClusterConstraints.ConstraintType;
import org.apache.helix.model.ConstraintItem;
import org.apache.helix.model.IdealState;
import org.apache.helix.model.InstanceConfig;
import org.apache.helix.model.StateModelDefinition;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

//import org.nd4j.paramserver.pistachio.pb.platform.perf.IncrementCounter;
//import org.nd4j.paramserver.pistachio.pb.platform.perf.InflightCounter;
// Generated code


public class PistachiosFormatter{

    public static class PistachioClusterInfo {
        List<String> hostList;
        int numPartitions;
        int numReplicas;
    }

    private static Logger logger = LoggerFactory.getLogger(PistachiosServer.class);

    public static PistachioClusterInfo getClusterInfo() {
        try {
            String zookeeperConnStr = ConfigurationManager.getConfiguration().getString("Pistachio.ZooKeeper.Server");
            ZKHelixAdmin admin = new ZKHelixAdmin(zookeeperConnStr);
            IdealState idealState = admin.getResourceIdealState("PistachiosCluster", "PistachiosResource");
            PistachioClusterInfo info = new PistachioClusterInfo();
            info.numPartitions = idealState.getNumPartitions();
            info.numReplicas = Integer.parseInt(idealState.getReplicas());
            info.hostList = admin.getInstancesInCluster("PistachiosCluster");

            logger.info("num partitions: {}, num Replicas: {}, hostList: {}.", info.numPartitions,
                    info.numReplicas, Joiner.on(",").join(info.hostList.toArray()));

            return info;
        } catch (Exception e) {
            logger.info("error getting cluster info", e);
            return null;
        }
    }

    private static void setConstraints(ZKHelixAdmin admin, int numPartitions) {

        logger.info("pause cluster");
        admin.enableCluster("PistachiosCluster", false);
        // setting partition constraints
        logger.info("setting per partition state transition constraints to 1");
        try {
            for (int constraintId = 0; constraintId < numPartitions; constraintId++) {
                java.util.HashMap<ConstraintAttribute, String>  attributes = new java.util.HashMap<ConstraintAttribute, String>();
                attributes.put(ConstraintAttribute.RESOURCE, "PistachiosResource");
                attributes.put(ConstraintAttribute.PARTITION, "PistachiosResource_"+constraintId);
                logger.info("setting per partition for {} state transition constraints to 1", "PistachiosResource_"+constraintId);
                admin.setConstraint("PistachiosCluster", ConstraintType.STATE_CONSTRAINT, "PistachiosPartitionTransitionConstraint" + constraintId,
                        new ConstraintItem(attributes,"1"));
            }

        } catch(Exception e) {
            logger.info("setting state transition constraints error, roll back and exit", e);
        }
        logger.info("resume cluster");
        admin.enableCluster("PistachiosCluster", true);
    }
    private static void format(ZKHelixAdmin admin, ZkClient zkClient, String[] hostList, int numPartitions,
                               int numReplicas, String kafkaTopicPrefix, String kafkaZKPath, String[] kafkaIds) {
        String [] kafkaReplicaAssignmentStrList = new String[numPartitions];
        try {
            //ZKHelixAdmin admin = new ZKHelixAdmin(args[1]);
            admin.addCluster("PistachiosCluster");

            // define model
            StateModelDefinition stateModel = new StateModelDefinition.Builder("MasterSlave")
                    // OFFLINE is the state that the system starts in (initial state is REQUIRED)
                    .initialState("OFFLINE")

                            // Lowest number here indicates highest priority, no value indicates lowest priority
                    .addState("MASTER", 1)
                    .addState("SLAVE", 2)
                    .addState("OFFLINE")

                            // Note the special inclusion of the DROPPED state (REQUIRED)
                    .addState(HelixDefinedState.DROPPED.toString())

                            // No more than one master allowed
                    .upperBound("MASTER", 1)

                            // R indicates an upper bound of number of replicas for each partition
                    .dynamicUpperBound("SLAVE", "R")

                            // Add some high-priority transitions
                    .addTransition("SLAVE", "MASTER", 1)
                    .addTransition("OFFLINE", "SLAVE", 2)

                            // Using the same priority value indicates that these transitions can fire in any order
                    .addTransition("MASTER", "SLAVE", 3)
                    .addTransition("SLAVE", "OFFLINE", 3)

                            // Not specifying a value defaults to lowest priority
                            // Notice the inclusion of the OFFLINE to DROPPED transition
                            // Since every state has a path to OFFLINE, they each now have a path to DROPPED (REQUIRED)
                    .addTransition("OFFLINE", HelixDefinedState.DROPPED.toString())

                            // Create the StateModelDefinition instance
                    .build();

            // Use the isValid() function to make sure the StateModelDefinition will work without issues
            // Assert.assertTrue(stateModel.isValid());
            try {
                admin.addStateModelDef("PistachiosCluster", "MasterSlave", stateModel);
            } catch (HelixException he) {
                if (he.getMessage().contains("already exists")) {
                    logger.info("state model def already exists, continue.", he);
                }
                else {
                    logger.info("adding state model def error, roll back and exit", he);
                    cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                    return;
                }
            } catch(Exception e) {
                logger.info("adding state model def error, roll back and exit", e);
                cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                return;
            }

            logger.info("adding resource");
            try {
                admin.addResource("PistachiosCluster", "PistachiosResource", numPartitions, "MasterSlave", "SEMI_AUTO");
            } catch (HelixException he) {
                if (he.getMessage().contains("already exists")) {
                    logger.info("resource already exists, continue.", he);
                }
                else {
                    logger.info("adding resource error, roll back and exit", he);
                    cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                    return;
                }
            } catch(Exception e) {
                logger.info("adding resource error, roll back and exit", e);
                cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                return;
            }
            logger.info("adding host");
            Map<String, String> hostToKafkaIdMap = new HashMap<>();
            int j = 0;
            for (String host : hostList) {
                InstanceConfig instanceConfig = new InstanceConfig(host);
                instanceConfig.setHostName(host);
                instanceConfig.setPort("1234");
                instanceConfig.setInstanceEnabled(true);
                hostToKafkaIdMap.put(host, kafkaIds[j++]);

                //Add additional system specific configuration if needed. These can be accessed during the node start up.
                //instanceConfig.getRecord().setSimpleField("key", "value");
                try {
                    admin.addInstance("PistachiosCluster", instanceConfig);
                } catch (HelixException he) {
                    if (he.getMessage().contains("already exists")) {
                        logger.info("instance already exists, continue.", he);
                    }
                    else {
                        logger.info("adding instance error, roll back and exit", he);
                        cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                        return;
                    }
                } catch(Exception e) {
                    logger.info("adding instance error, roll back and exit", e);
                    cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                    return;
                }
            }
            logger.info("rebalancing");
            try {
                admin.rebalance("PistachiosCluster", "PistachiosResource", numReplicas);
                // form kafka preference list according to helix preferrence list
                for (int i = 0; i < numPartitions; i++) {
                    IdealState idealState = admin.getResourceIdealState("PistachiosCluster", "PistachiosResource");
                    List<String> preferenceList = idealState.getPreferenceList("PistachiosResource_" + i);
                    logger.info("preference list for {} is {}", "PistachiosResource_" + i, org.apache.commons.lang3.StringUtils.join(preferenceList, ","));

                    kafkaReplicaAssignmentStrList[i] = "";
                    for (String preferredHost : preferenceList) {
                        if (kafkaReplicaAssignmentStrList[i].length() > 0)
                            kafkaReplicaAssignmentStrList[i] += ":";
                        kafkaReplicaAssignmentStrList[i] += hostToKafkaIdMap.get(preferredHost);
                    }
                    logger.info("kafka preference id list for {} is {}", "PistachiosResource_" + i, kafkaReplicaAssignmentStrList[i]);
                }
            } catch(Exception e) {
                logger.info("rebalancing error, roll back and exit", e);
                cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                return;
            }

            for (int i = 0; i < numPartitions; i++) {
                try {
                    logger.info("creating topic for {} with preference list {}", "PistachiosResource_" + i, kafkaReplicaAssignmentStrList[i]);
                    //CreateTopicCommand.createTopic(zkClient , "PistachiosPartition." + i, 1, 1, "");
                    CreateTopicCommand.createTopic(zkClient , kafkaTopicPrefix + i, 1, numReplicas, kafkaReplicaAssignmentStrList[i]);
                } catch (kafka.common.TopicExistsException tee) {
                    logger.info("topic exists, continue", tee);
                } catch (Exception e) {
                    logger.info("creating kafka topic error, roll back", e);
                    cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                    return;
                }
            }
            zkClient.close();


            // setting partition constraints
            logger.info("setting per partition state transition constraints to 1");
            try {
                for (int constraintId = 0; constraintId < numPartitions; constraintId++) {
                    java.util.HashMap<ConstraintAttribute, String>  attributes = new java.util.HashMap<ConstraintAttribute, String>();
                    attributes.put(ConstraintAttribute.RESOURCE, "PistachiosResource");
                    attributes.put(ConstraintAttribute.PARTITION, "PistachiosResource_"+constraintId);
                    logger.info("setting per partition for {} state transition constraints to 1", "PistachiosResource_"+constraintId);
                    admin.setConstraint("PistachiosCluster", ConstraintType.STATE_CONSTRAINT, "PistachiosPartitionTransitionConstraint" + constraintId,
                            new ConstraintItem(attributes,"1"));
                }

            } catch(Exception e) {
                logger.info("setting state transition constraints error, roll back and exit", e);
                cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
                return;
            }

            logger.info("adding topic to zk path: {}", kafkaZKPath);
            //ZkClient zkClient = new ZkClient(args[1]+ (kafkaZKPath != null ? "/" + kafkaZKPath : ""), 30000, 30000, ZKStringSerializer$.MODULE$);

            //Thread.sleep(120000);
            //
            logger.info("format finished succeessfully");
        } catch(Exception e) {
            logger.info("error:", e);
        }
    }

    private static void cleanup(ZKHelixAdmin admin, ZkClient zkClient, String[] hostList, int numPartitions, int numReplicas, String kafkaTopicPrefix, String kafkaZKPath) {
        try {
            // TODO, delete not supported until 0.8.1, we'll enable it later
            for (int i =0; i<numPartitions; i++) {
                //zkClient = new ZkClient(zkConnect, 30000, 30000, ZKStringSerializer);
                zkClient.deleteRecursive(ZkUtils.getTopicPath(kafkaTopicPrefix + i));
            }
            zkClient.close();
            //ZKHelixAdmin admin = new ZKHelixAdmin(args[1]);
            admin.dropCluster("PistachiosCluster");
        } catch(Exception e) {
            logger.info("error:", e);
        }
        logger.info("cleanup finished succeessfully");
    }

    private static void handleFormatCommand(String [] args) {
        String zookeeperConnStr = ConfigurationManager.getConfiguration().getString("Pistachio.ZooKeeper.Server");
        String kafkaTopicPrefix = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix");

        ZKHelixAdmin admin = null;
        ZkClient zkClient = null;

        Options options = new Options();
        options.addOption(Option.builder("h")
                .longOpt("hostlist")
                .required(true)
                .desc("comma separated list of hosts of the cluster")
                .hasArg()
                .type(new String().getClass()).build());
        options.addOption(Option.builder("p")
                .longOpt("partitionnumber")
                .required(true)
                .desc("number of partitions")
                .hasArg()
                .type(new Integer(0).getClass()).build());
        options.addOption(Option.builder("r")
                .longOpt("replicanumber")
                .required(true)
                .desc("number of replicas")
                .hasArg()
                .type(new Integer(0).getClass()).build());
        options.addOption(Option.builder("k")
                .longOpt("kafkaidlist")
                .required(true)
                .desc("comma separated list of the kafka ids in the order of the host list, the command will make sure kafka and pistachio master and slaves will be co-located to reduce the network transfer")
                .hasArg()
                .type(new String().getClass()).build());
        options.addOption(Option.builder("z")
                .longOpt("kafkazkpath")
                .required(false)
                .desc("the zk chroot path for kafka")
                .hasArg()
                .type(new String().getClass()).build());

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;

        try {
            logger.info("parsing {}", Arrays.deepToString(args));
            cmd = parser.parse( options, args);
            try {
                admin = new ZKHelixAdmin(zookeeperConnStr);
                zkClient = new ZkClient(zookeeperConnStr + (cmd.hasOption("kafkazkpath")? "/" + cmd.getOptionValue("kafkazkpath"): ""),
                        30000, 30000, ZKStringSerializer$.MODULE$);
                int numPartitions = Integer.parseInt(cmd.getOptionValue("partitionnumber"));
                int numReplicas = Integer.parseInt(cmd.getOptionValue("replicanumber"));

                logger.info("calling format with parameters: hostlist {}, num partition {}, num replicas {}, kafka prefix {}, kafka path {}, kafka id list {}",
                        cmd.getOptionValues("hostlist"), numPartitions, numReplicas, kafkaTopicPrefix, cmd.getOptionValue("kafkazkpath"),
                        cmd.getOptionValues("kafkaidlist"));
                format(admin, zkClient, cmd.getOptionValue("hostlist").split(","), numPartitions, numReplicas,
                        kafkaTopicPrefix, (cmd.hasOption("kafkazkpath")? "/" + cmd.getOptionValue("kafkazkpath"): ""),
                        cmd.getOptionValue("kafkaidlist").split(","));
            } catch(Exception e) {
                logger.info("error:", e);
            } finally {
                if (zkClient != null)
                    zkClient.close();
            }
        } catch (Exception e) {
            System.out.println("Error parsing: \n" + e.getMessage() + "\n");

            String header = "This command is for formating a pistachio cluster.\n\n";
            String footer = "Any issues pls contact .\n";

            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("format_cluster.sh", header, options, footer, true);
        }


    }



    private static void handleCleanupCommand(String [] args) {
        String zookeeperConnStr = ConfigurationManager.getConfiguration().getString("Pistachio.ZooKeeper.Server");
        String kafkaTopicPrefix = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix");

        ZKHelixAdmin admin = null;
        ZkClient zkClient = null;

        Options options = new Options();
        options.addOption(Option.builder("h")
                .longOpt("hostlist")
                .required(true)
                .desc("comma seperated list of hosts of the cluster")
                .hasArg()
                .type(new String().getClass()).build());
        options.addOption(Option.builder("p")
                .longOpt("partitionnumber")
                .required(true)
                .desc("number of partitions")
                .hasArg()
                .type(new Integer(0).getClass()).build());
        options.addOption(Option.builder("r")
                .longOpt("replicanumber")
                .required(true)
                .desc("number of replicas")
                .hasArg()
                .type(new Integer(0).getClass()).build());
        options.addOption(Option.builder("z")
                .longOpt("kafkazkpath")
                .required(false)
                .desc("the zk chroot path for kafka")
                .hasArg()
                .type(new String().getClass()).build());


        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;

        try {
            cmd = parser.parse( options, args);
        } catch (Exception e) {
            System.out.println("Error parsing: \n" + e.getMessage() + "\n");

            String header = "This command is for formating a pistachio cluster.\n\n";
            String footer = "Any issues pls contact .\n";

            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("format_cluster.sh", header, options, footer, true);
            return;
        }

        try {
            admin = new ZKHelixAdmin(zookeeperConnStr);
            zkClient = new ZkClient(zookeeperConnStr + (cmd.hasOption("kafkazkpath")? "/" + cmd.getOptionValue("kafkazkpath"): ""),
                    30000, 30000, ZKStringSerializer$.MODULE$);
            int numPartitions = Integer.parseInt(cmd.getOptionValue("partitionnumber"));
            int numReplicas = Integer.parseInt(cmd.getOptionValue("replicanumber"));

            cleanup(admin, zkClient, cmd.getOptionValue("hostlist").split(","), numPartitions, numReplicas,
                    kafkaTopicPrefix, (cmd.hasOption("kafkazkpath")? "/" + cmd.getOptionValue("kafkazkpath"): ""));
        } catch(Exception e) {
            logger.info("error:", e);
        } finally {
            zkClient.close();
        }

    }

    public static void main(String [] args) {

        String usage = "Usage: \tformat_cluster.sh commands options\n";
        usage += "\t commands can be one of info, format, cleanup. Type a command to see usage for it.\n";


        if (args.length < 1) {
            System.out.println("invalid number of parameters");
            System.out.println(usage);
            return;
        } else if ("info".equals(args[0])) {
            getClusterInfo();
            return;
        } else if ("format".equals(args[0])) {
            handleFormatCommand(args.length >1 ? Arrays.copyOfRange(args, 1, args.length ): null);
        } else if ("cleanup".equals(args[0])) {
            handleCleanupCommand(args.length >1 ? Arrays.copyOfRange(args, 1, args.length): null);
        } else {
            System.out.println("invalid number of parameters");
            System.out.println(usage);
            return;
        }

    }
  /*
  public static void main0(String [] args) {
      String usage = "Usage: \tformat_cluster.sh info \n";
      usage += "\tformat_cluster.sh format [comma seperated cluster] [num partition] [num replica] [comma seperated kafka ids of the corresponding pistachio server] (kafka zk path, optional) \n";
      usage += "\tformat_cluster.sh cleanup [comma seperated cluster] [num partition] [num replica] (kafka zk path, optional) \n";
      //usage += "\tformat_cluster.sh set_constraints [comma seperated cluster] [num partition] \n";
      usage += "\tkafkaTopicPrefix, ZK server conn string should also be set from config. \n";

      if (args.length > 0 && "info".equals(args[0])) {
          getClusterInfo();
          return;
      }

      if (args.length <4) {
          System.out.println("invalid number of parameters");
          System.out.println(usage);
          return;
      }

      int numPartitions = -1;
      int numReplicas = -1;
      String[] hostList = null;
      String kafkaZKPath = null;
      String kafkaTopicPrefix = null;

      boolean cleanup = false;
      boolean setConstraints = false;

      try {
          numPartitions = Integer.parseInt(args[2]);
          numReplicas = Integer.parseInt(args[3]);
          if ("cleanup".equals(args[0])) {
              cleanup = true;
          }
          if ("set_constraints".equals(args[0])) {
              setConstraints = true;
          }
          hostList = args[1].split(",");
          if (args.length >=5)
              kafkaZKPath = args[4];

      } catch(Exception e) {
      }

      String zookeeperConnStr = ConfigurationManager.getConfiguration().getString("Pistachio.ZooKeeper.Server");
      kafkaTopicPrefix = ConfigurationManager.getConfiguration().getString("Profile.Kafka.TopicPrefix");

      if (numPartitions == -1 || numReplicas == -1|| hostList == null || hostList.length == 0 || kafkaTopicPrefix == null || zookeeperConnStr == null) {
          System.out.println("invalid parameters");
          System.out.println(usage);
          return;
      }





    try {
        ZKHelixAdmin admin = new ZKHelixAdmin(zookeeperConnStr);

        ZkClient zkClient = new ZkClient(zookeeperConnStr + (kafkaZKPath != null ? "/" + kafkaZKPath : ""), 30000, 30000, ZKStringSerializer$.MODULE$);

        if (setConstraints) {
            setConstraints(admin, numPartitions);

        } else if (cleanup) {
            cleanup(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
        } else {
            format(admin, zkClient, hostList, numPartitions, numReplicas, kafkaTopicPrefix, kafkaZKPath);
        }

        zkClient.close();
        //Thread.sleep(120000);
      } catch(Exception e) {
          logger.info("error:", e);
      }
  }
  */

}
