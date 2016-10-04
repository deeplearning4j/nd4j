package org.nd4j.parameterserver.client;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.aeron.ipc.response.AeronNDArrayResponder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class ParameterServerClientTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(ParameterServerClientTest.class);
    private Aeron.Context ctx;
    private Aeron.Context ctx2;

    @Before
    public void before() {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());
        mediaDriver = MediaDriver.launchEmbedded(ctx);
        System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
        System.out.println("Launched media driver");
    }



    @Test
    public void testServer() throws Exception {
        int streamId = 10;
        int responderStreamId = 11;
        int responderPort = 40124;
        int subscriberPort = 40123;
        String host = "127.0.0.1";
        AeronNDArrayResponder responder = AeronNDArrayResponder.startSubscriber(
                getContext2(),
                host,
                responderPort,
                (NDArrayHolder) () -> Nd4j.scalar(1.0)
                ,responderStreamId);

        AtomicInteger count = new AtomicInteger(0);

        AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.startSubscriber(
                getContext(),
                host,
                subscriberPort,
                arr -> count.incrementAndGet()
                ,streamId);

        Thread.sleep(10000);

        ParameterServerClient client = ParameterServerClient
                 .builder()
                 .ctx(getContext2())
                 .ndarrayRetrieveUrl(responder.connectionUrl())
                 .ndarraySendUrl(subscriber.connectionUrl())
                 .subscriberHost("localhost")
                 .subscriberPort(40125)
                 .subscriberStream(11).build();

        client.pushNDArray(Nd4j.scalar(1.0));
        INDArray arr = client.getArray();
        assertEquals(Nd4j.scalar(1.0),arr);
    }

    private Aeron.Context getContext2() {
        if(ctx2 == null)
            ctx2 = new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
        return ctx2;
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
