package org.nd4j.paramserver.pistachio.pb.mttf;

import org.junit.Test;
import org.nd4j.paramserver.pistachio.pb.PistachiosClient;
import org.nd4j.paramserver.pistachio.pb.PistachiosFormatter;
import org.nd4j.paramserver.pistachio.pb.PistachiosServer;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedKafkaCluster;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedZooKeeper;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.ZooKeeperRunner;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;
import java.util.Properties;

/**
 * @author Adam Gibson
 */
public class SelfContainedTest {
    @Test
    public void testStart() throws Exception {
        final EmbeddedZooKeeper zk = new EmbeddedZooKeeper();
        final EmbeddedKafkaCluster cluster = new EmbeddedKafkaCluster(zk.getConnection(),new Properties(), Arrays.asList(9090));

        zk.startup();

        cluster.startup();


        ClassPathResource resource = new ClassPathResource("config");
        System.setProperty("configPath", resource.getFile().getAbsolutePath());

        System.setProperty("Pistachio.ZooKeeper.Server", zk.getConnection());


        PistachiosFormatter.main(new String[]{"format", "-h", "127.0.0.1", "-k", "1", "-p", "1", "-r", "1"});

        PistachiosServer.main(new String[]{});

        int threadCount = 50;


        PistachiosClient client = new PistachiosClient();
        String hi = "hi";
        client.store(hi.getBytes(),hi.getBytes());
        byte[] b = client.lookup(hi.getBytes(),true);
        System.out.println(new String(b));
        Thread.sleep(10000);


    }


}