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
import org.nd4j.paramserver.pistachio.pb.PistachiosClient;
import org.nd4j.paramserver.pistachio.pb.PistachiosFormatter;
import org.nd4j.paramserver.pistachio.pb.PistachiosServer;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedKafkaCluster;
import org.nd4j.paramserver.pistachio.pb.kafka.embedded.EmbeddedZooKeeper;
import org.nd4j.paramserver.pistachio.pb.util.NativeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;


public class PistachiosMTTFTest extends Thread {
    private static Logger logger = LoggerFactory.getLogger(PistachiosMTTFTest.class);
    private static AtomicInteger successCounter = new AtomicInteger(0);
    private static AtomicInteger failureCounter = new AtomicInteger(0);
    private static int threadNumber = 10;
    private static PistachiosClient client;
    private static Random rand = new Random();

    public static void init() {
        try {
            client = new PistachiosClient();
        }catch (Exception e) {
            logger.info("error creating client", e);
            System.exit(0);
        }
    }

    public static synchronized  PistachiosClient getClient() {
        return client;
    }


    public static void main(String [] args) throws Exception {
        final EmbeddedZooKeeper zk = new EmbeddedZooKeeper(2181);
        final EmbeddedKafkaCluster cluster = new EmbeddedKafkaCluster(zk.getConnection(),new Properties(), Arrays.asList(9090));

        zk.startup();

        cluster.startup();


        ClassPathResource resource = new ClassPathResource("config");
        System.setProperty("configPath", resource.getFile().getAbsolutePath());

        System.setProperty("Pistachio.ZooKeeper.Server", zk.getConnection());


        PistachiosFormatter.main(new String[]{"format", "-h", "127.0.0.1", "-k", "1", "-p", "1", "-r", "1"});

        PistachiosServer.main(new String[]{});


        init();
        while(true) {
            try {
                long id = rand.nextLong();
                //String value=InetAddress.getLocalHost().getHostName() + rand.nextInt() ;
                String value=NativeUtils.getHostname() + Thread.currentThread().getId() + rand.nextInt() ;
                getClient().store(com.google.common.primitives.Longs.toByteArray(id), value.getBytes());
                Thread.sleep(1000);
                for (int i =0 ; i < 30; i++) {
                    byte[] clientValue = getClient().lookup(com.google.common.primitives.Longs.toByteArray(id), true);
                    if(clientValue == null) {
                        System.out.println(clientValue);
                        break;
                    }
                    String remoteValue = new String(clientValue);
                    //if (Arrays.equals(value.getBytes(), clientValue) || !remoteValue.contains(InetAddress.getLocalHost().getHostName())) {
                    if (Arrays.equals(value.getBytes(), clientValue) ||
                            !remoteValue.contains(NativeUtils.getHostname() + Thread.currentThread().getId()  )) {
                        logger.debug("succeeded checking id {} value {}, {}/{}", id, value, failureCounter.get(), successCounter.get());
                    } else {
                        logger.error("failed checking id {} value {} != {}", id, value, new String(clientValue));
                        failureCounter.incrementAndGet();
                        break;
                        //System.exit(0);
                    }
                    Thread.sleep(100);
                }
                successCounter.incrementAndGet();

                if (successCounter.get() % 10 == 0) {
                    System.out.println("testing result: "+ failureCounter.get()+"/"+ successCounter.get());
                }
            } catch (Exception e) {
                System.out.println("error testing"+ e);
                System.exit(0);
            }
        }
    }
    public void run() {

        while(true) {
            try {
                long id = rand.nextLong();
                //String value=InetAddress.getLocalHost().getHostName() + rand.nextInt() ;
                String value=NativeUtils.getHostname() + Thread.currentThread().getId() + rand.nextInt() ;
                getClient().store(com.google.common.primitives.Longs.toByteArray(id), value.getBytes());
                Thread.sleep(1000);
                for (int i =0 ; i < 30; i++) {
                    byte[] clientValue = getClient().lookup(com.google.common.primitives.Longs.toByteArray(id), true);
                    if(clientValue == null) {
                        System.out.println(clientValue);
                        break;
                    }
                    String remoteValue = new String(clientValue);
                    //if (Arrays.equals(value.getBytes(), clientValue) || !remoteValue.contains(InetAddress.getLocalHost().getHostName())) {
                    if (Arrays.equals(value.getBytes(), clientValue) ||
                            !remoteValue.contains(NativeUtils.getHostname() + Thread.currentThread().getId()  )) {
                        logger.debug("succeeded checking id {} value {}, {}/{}", id, value, failureCounter.get(), successCounter.get());
                    } else {
                        logger.error("failed checking id {} value {} != {}", id, value, new String(clientValue));
                        failureCounter.incrementAndGet();
                        break;
                        //System.exit(0);
                    }
                    Thread.sleep(100);
                }
                successCounter.incrementAndGet();

                if (successCounter.get() % 10 == 0) {
                    System.out.println("testing result: "+ failureCounter.get()+"/"+ successCounter.get());
                }
            } catch (Exception e) {
                System.out.println("error testing"+ e);
                System.exit(0);
            }
        }
    }

}
