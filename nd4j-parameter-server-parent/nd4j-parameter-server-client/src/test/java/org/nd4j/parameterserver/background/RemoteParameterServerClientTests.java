package org.nd4j.parameterserver.background;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.client.ParameterServerClient;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/5/16.
 */
@Slf4j
public class RemoteParameterServerClientTests {
    private int parameterLength = 1000;
    private Aeron.Context ctx;
    private MediaDriver mediaDriver;

    @Before
    public void before() throws Exception {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());

        mediaDriver = MediaDriver.launchEmbedded(ctx);


        Thread t = new Thread(() -> {
            try {
                BackgroundDaemonStarter.startMaster(parameterLength,mediaDriver.aeronDirectoryName());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        t.start();
        log.info("Started master");
        Thread t2 = new Thread(() -> {
            try {
                BackgroundDaemonStarter.startSlave(parameterLength,mediaDriver.aeronDirectoryName());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        t2.start();
        log.info("Started slave");

    }


    @After
    public void after() throws Exception {
        mediaDriver.close();
    }

    @Test
    public void remoteTests() throws Exception {
        ParameterServerClient client = ParameterServerClient
                .builder()
                .ctx(getContext())
                .ndarrayRetrieveUrl(BackgroundDaemonStarter.masterResponderUrl())
                .ndarraySendUrl(BackgroundDaemonStarter.slaveConnectionUrl())
                .subscriberHost("localhost")
                .masterStatusHost("localhost")
                .masterStatusPort(9999)
                .subscriberPort(40125)
                .subscriberStream(12).build();

        assertEquals("localhost:40125:12",client.connectionUrl());
        while(!client.masterStarted())
            Thread.sleep(1000);


        //flow 1:
        /**
         * Client (40125:12): sends array to listener on slave(40126:10)
         * which publishes to master (40123:11)
         * which adds the array for parameter averaging.
         * In this case totalN should be 1.
         */
        client.pushNDArray(Nd4j.ones(parameterLength));
        Thread.sleep(1000);
        log.info("Pushed ndarray");
        INDArray arr = client.getArray();
        assertEquals(Nd4j.ones(1000),arr);

/*
        StopWatch stopWatch = new StopWatch();
        long nanoTimeTotal = 0;
        int n = 1000;
        for(int i = 0; i < n; i++) {
            stopWatch.start();
            client.getArray();
            stopWatch.stop();
            nanoTimeTotal += stopWatch.getNanoTime();
            stopWatch.reset();
        }

        System.out.println(nanoTimeTotal / n);
*/



    }





    private Aeron.Context getContext() {
        if(ctx == null)
            ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

}
