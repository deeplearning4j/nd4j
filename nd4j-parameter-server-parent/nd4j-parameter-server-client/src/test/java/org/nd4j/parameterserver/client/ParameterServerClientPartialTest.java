package org.nd4j.parameterserver.client;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.ParameterServerListener;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class ParameterServerClientPartialTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(ParameterServerClientPartialTest.class);
    private Aeron.Context ctx;
    private ParameterServerSubscriber masterNode,slaveNode;
    private int[] shape = {2,2};

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
        masterNode = new ParameterServerSubscriber(mediaDriver);
        masterNode.run(new String[] {
                "-m","true",
                "-p","40123",
                "-h","localhost",
                "-id","11",
                "-md", mediaDriver.aeronDirectoryName(),
                "-sp", "10000",
                "-s","2,2"
        });

        assertTrue(masterNode.isMaster());
        assertEquals(1000,masterNode.getParameterLength());
        assertEquals(40123,masterNode.getPort());
        assertEquals("localhost",masterNode.getHost());
        assertEquals(11,masterNode.getStreamId());
        assertEquals(12,masterNode.getResponder().getStreamId());
        assertEquals(masterNode.getMasterArray(),Nd4j.create(new int[]{2,2}));

        slaveNode = new ParameterServerSubscriber(mediaDriver);
        slaveNode.run(new String[] {
                "-p","40126",
                "-h","localhost",
                "-id","10",
                "-pm",masterNode.getSubscriber().connectionUrl(),
                "-md", mediaDriver.aeronDirectoryName(),
                "-sp", "11000"
        });

        assertFalse(slaveNode.isMaster());
        assertEquals(1000,slaveNode.getParameterLength());
        assertEquals(40126,slaveNode.getPort());
        assertEquals("localhost",slaveNode.getHost());
        assertEquals(10,slaveNode.getStreamId());

        int tries = 10;
        while(!masterNode.subscriberLaunched() && !slaveNode.subscriberLaunched() && tries < 10) {
            Thread.sleep(10000);
            tries++;
        }

        if(!masterNode.subscriberLaunched() && !slaveNode.subscriberLaunched()) {
            throw new IllegalStateException("Failed to start master and slave node");
        }

        log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        log.info("Launched media driver");
    }



    @Test
    public void testServer() throws Exception {
        ParameterServerClient client = ParameterServerClient
                .builder()
                .ctx(getContext())
                .ndarrayRetrieveUrl(masterNode.getResponder().connectionUrl())
                .ndarraySendUrl(slaveNode.getSubscriber().connectionUrl())
                .subscriberHost("localhost")
                .subscriberPort(40125)
                .subscriberStream(12).build();
        assertEquals("localhost:40125:12",client.connectionUrl());
        //flow 1:
        /**
         * Client (40125:12): sends array to listener on slave(40126:10)
         * which publishes to master (40123:11)
         * which adds the array for parameter averaging.
         * In this case totalN should be 1.
         */
        client.pushNDArrayMessage(NDArrayMessage.of(Nd4j.ones(2),new int[]{0},0));
        log.info("Pushed ndarray");
        Thread.sleep(10000);
        ParameterServerListener listener = (ParameterServerListener) masterNode.getCallback();
        assertEquals(1,listener.getTotalN().get());
        INDArray assertion = Nd4j.create(new int[]{2,2});
        assertion.getColumn(0).addi(1.0);
        assertEquals(assertion,listener.getArr());
        INDArray arr = client.getArray();
        assertEquals(assertion,arr);
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
