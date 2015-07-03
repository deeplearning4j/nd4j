package org.nd4j.paramserver.pistachio.pb.kafka.embedded;

import kafka.server.KafkaConfig;
import kafka.server.KafkaServer;
import kafka.server.KafkaServerStartable;

import java.util.Properties;

/**
 * @author Adam Gibson
 */
public class EmbeddedKafka {
    private String hostName = "127.0.0.1";
    private int port = 9090;
    private int brokerId = 1;
    private String zooKeeperHost = "localhost:2181";
    private String brokerList = "localhost:9092";
    private KafkaServerStartable server;
    public void start() {
        Properties props = new Properties();
        props.setProperty("hostname",hostName);
        props.setProperty("port", String.valueOf(port));
        props.setProperty("log.dir", "/tmp/embeddedkafka/");
        props.setProperty("enable.zookeeper", "true");
        props.put("zookeeper.connect", zooKeeperHost);
        props.put("broker.id", String.valueOf(brokerId));
        props.put("metadata.broker.list",brokerList);




        final ZooKeeperRunner runner = new ZooKeeperRunner();
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    runner.run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).start();

        server = new KafkaServerStartable(new KafkaConfig(props));
        server.startup();

        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                server.shutdown();
                server.awaitShutdown();
            }
        }));
    }

    public String getHostName() {
        return hostName;
    }

    public void setHostName(String hostName) {
        this.hostName = hostName;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public int getBrokerId() {
        return brokerId;
    }

    public void setBrokerId(int brokerId) {
        this.brokerId = brokerId;
    }


    public String getZooKeeperHost() {
        return zooKeeperHost;
    }

    public void setZooKeeperHost(String zooKeeperHost) {
        this.zooKeeperHost = zooKeeperHost;
    }

    public String getBrokerList() {
        return brokerList;
    }

    public void setBrokerList(String brokerList) {
        this.brokerList = brokerList;
    }
}
