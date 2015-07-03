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

// Generated code
//import tutorial.*;
//import shared.*;

package org.nd4j.paramserver.pistachio.pb.mttf;
import kyotocabinet.DB;
import org.nd4j.paramserver.pistachio.pb.PistachiosClient;
import org.nd4j.paramserver.pistachio.pb.PistachiosFormatter;
import org.nd4j.paramserver.pistachio.pb.PistachiosServer;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedKafkaCluster;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.ZooKeeperRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Random;


public class PistachiosBenchmarking implements Runnable {
    private static Logger logger = LoggerFactory.getLogger(PistachiosBenchmarking.class);
    private PistachiosClient client;
    private Random rand;
    private static int recordAverageSize = 1024;

    public PistachiosBenchmarking() {
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

        final ZooKeeperRunner zk = new ZooKeeperRunner();
        final EmbeddedKafkaCluster cluster = new EmbeddedKafkaCluster(zk.toString());
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    zk.run();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
        t.start();

        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                cluster.startup();

            }
        });

        t2.start();


        Thread.sleep(10000);
        ClassPathResource resource = new ClassPathResource("config");
        System.setProperty("configPath", resource.getFile().getAbsolutePath());

        System.setProperty("Pistachio.ZooKeeper.Server", zk.toString());


        PistachiosFormatter.main(new String[]{"format", "-h", "127.0.0.1", "-k", "1", "-p", "1", "-r", "1"});

        PistachiosServer.main(new String[]{});

        int threadCount = 50;

        PistachiosClient client = new PistachiosClient();
        String hi = "hi";
        client.store(hi.getBytes(),hi.getBytes());
        byte[] b = client.lookup(hi.getBytes());
        System.out.println(new String(b));
        Thread.sleep(10000);


    }

}
