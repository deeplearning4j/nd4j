package org.nd4j.aeron.ipc.response;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.*;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class AeronNDArrayReponseTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(NdArrayIpcTest.class);
    private Aeron.Context ctx;

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
    public void testResponse() throws Exception {
        int streamId = 10;
        int responderStreamId = 11;
        String host = "127.0.0.1";
        AeronNDArrayResponder.startSubscriber(
                getContext(),
                host,
                40123,
                (NDArrayHolder) () -> Nd4j.scalar(1.0)
                ,responderStreamId);

        AtomicInteger count = new AtomicInteger(0);

        AeronNDArraySubscriber.startSubscriber(
                getContext(),
                host,
                40124,
                arr -> count.incrementAndGet()
                ,streamId);


        Thread.sleep(10000);

        int expectedResponses = 10;
        HostPortPublisher publisher = HostPortPublisher.builder()
                .uriToSend(host + ":40124")
                .channel(AeronUtil.aeronChannel(host,40123))
                .streamId(streamId).build();
        for(int i = 0; i < expectedResponses; i++) {
            publisher.send();
        }

        publisher.close();

        assertEquals(expectedResponses,count.get());

    }

    private Aeron.Context getContext() {
        if(ctx == null)
            ctx = new Aeron.Context().publicationConnectionTimeout(-1).availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

}
