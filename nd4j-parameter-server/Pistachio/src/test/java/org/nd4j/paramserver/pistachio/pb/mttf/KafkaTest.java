package org.nd4j.paramserver.pistachio.pb.mttf;

import kafka.admin.CreateTopicCommand;
import kafka.api.FetchRequest;
import kafka.api.FetchRequestBuilder;
import kafka.consumer.ConsumerConfig;
import kafka.javaapi.FetchResponse;
import kafka.javaapi.consumer.SimpleConsumer;
import kafka.javaapi.message.ByteBufferMessageSet;
import kafka.javaapi.producer.Producer;
import kafka.message.MessageAndOffset;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;
import kafka.utils.ZKStringSerializer$;
import org.apache.helix.manager.zk.ZkClient;
import org.junit.Test;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedKafkaCluster;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedZooKeeper;
import org.testng.internal.Utils;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

/**
 * @author Adam Gibson
 */
public class KafkaTest {
    @Test
    public void testSend() throws Exception {
        long events = 10;
        Random rnd = new Random();
        Properties props = new Properties();

        props.put("metadata.broker.list", "127.0.0.1:9090");
        props.put("serializer.class", "kafka.serializer.StringEncoder");
        props.put("request.required.acks", "1");

        final EmbeddedZooKeeper zk = new EmbeddedZooKeeper(2181);
        final EmbeddedKafkaCluster cluster = new EmbeddedKafkaCluster(zk.getConnection(), props, Arrays.asList(9090));
        zk.startup();
        cluster.startup();
        Thread.sleep(10000);


        ProducerConfig config = new ProducerConfig(props);

        Producer<String, String> producer = new Producer<>(config);


        String topic = "page_visits";
        for (long nEvents = 0; nEvents < events; nEvents++) {
            long runtime = new Date().getTime();
            String ip = "192.168.2." + rnd.nextInt(255);
            String msg = runtime + ",www.example.com," + ip;
            KeyedMessage<String, String> data = new KeyedMessage<>(topic, ip, msg);
            producer.send(data);
        }

        ZkClient zkClient = new ZkClient(zk.getConnection(), 30000, 30000, ZKStringSerializer$.MODULE$);

        //CreateTopicCommand.createTopic(zkClient, topic, 1, 1, "");

        HelloKafkaConsumer consumer = new HelloKafkaConsumer(topic,zk.getConnection());
        consumer.start();

        Thread.sleep(30000);



    }





}