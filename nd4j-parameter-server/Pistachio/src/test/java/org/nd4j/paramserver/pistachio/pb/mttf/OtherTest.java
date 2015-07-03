package org.nd4j.paramserver.pistachio.pb.mttf;

import org.nd4j.paramserver.pistachio.pb.PistachiosClient;
import org.nd4j.paramserver.pistachio.pb.PistachiosFormatter;
import org.nd4j.paramserver.pistachio.pb.PistachiosServer;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedKafkaCluster;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedZooKeeper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;
import java.util.Properties;
import java.util.Random;

/**
 * @author Adam Gibson
 */
public class OtherTest implements Runnable {

    private static Logger logger = LoggerFactory.getLogger(PistachiosBenchmarking.class);
    private PistachiosClient client;
    private Random rand;
    private static int recordAverageSize = 1024;


    public OtherTest() {
        try {
            client = new PistachiosClient();
        }catch (Exception e) {
            logger.info("error creating client", e);
            return;
        }
        rand = new Random();
    }
    public void run() {
        logger.info("starting thread....");
        while(true) {
            try {
                long id = rand.nextLong();
                byte[] bytes = new byte[rand.nextInt(recordAverageSize << 1)];
                rand.nextBytes(bytes);
                client.store(com.google.common.primitives.Longs.toByteArray(id), bytes);
            } catch (Exception e) {
                System.out.println("error testing"+ e);
                System.exit(0);
            }
        }
    }


    public static void main(String [] args) throws Exception {
        int threadCount = 50;
        logger.info("parsing error {}", args.length);
        final EmbeddedZooKeeper zk = new EmbeddedZooKeeper();
        final EmbeddedKafkaCluster cluster = new EmbeddedKafkaCluster(zk.getConnection(),new Properties(), Arrays.asList(9090));

        zk.startup();

        cluster.startup();


        ClassPathResource resource = new ClassPathResource("config");
        System.setProperty("configPath", resource.getFile().getAbsolutePath());

        System.setProperty("Pistachio.ZooKeeper.Server", zk.getConnection());


        PistachiosFormatter.main(new String[]{"format", "-h", "127.0.0.1", "-k", "1", "-p", "1", "-r", "1"});

        PistachiosServer.main(new String[]{});


        if (args.length >=1) {
            try {
                threadCount = Integer.parseInt(args[0]);
                logger.info("parsd {} {}", args[0], threadCount);
            } catch(Exception e) {
                logger.info("parsing error", e);
            }
        }
        if (args.length >=2) {
            try {
                recordAverageSize = Integer.parseInt(args[1]);
                logger.info("parsd {} {}", args[1], recordAverageSize);
            } catch(Exception e) {
                logger.info("parsing error", e);
            }
        }

        for (int i = 0; i< threadCount; i++) {
            Thread thread = new Thread(new OtherTest(), "benchmarking"+i);
            thread.start();
        }

    }



}