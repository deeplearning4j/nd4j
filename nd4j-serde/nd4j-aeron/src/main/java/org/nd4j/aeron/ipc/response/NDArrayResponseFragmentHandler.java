package org.nd4j.aeron.ipc.response;

import io.aeron.Aeron;
import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.agrona.DirectBuffer;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A subscriber that listens for host port pairs.
 * These are meant to be aeron channels.
 *
 * Given an @link{NDArrayHolder} it will send
 * the ndarray to the designated channel by the subscriber.
 */
@AllArgsConstructor
@Builder
public class NDArrayResponseFragmentHandler implements FragmentHandler {
    private NDArrayHolder holder;
    private Aeron.Context context;
    private Aeron aeron;
    private int streamId;

    /**
     * Callback for handling fragments of data being read from a log.
     *
     * @param buffer containing the data.
     * @param offset at which the data begins.
     * @param length of the data in bytes.
     * @param header representing the meta data for the data.
     */
    @Override
    public void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        String hostPort = new String(buffer.byteArray());
        String[] split = hostPort.split(":");
        if(split == null || split.length != 2)
            throw new IllegalStateException("Illegal string input " + hostPort);
        int port = Integer.parseInt(split[1]);
        String channel = String.format("aeron:udp?endpoint=%s:%d",split[0],port);
        INDArray arrGet = holder.get();
        try(AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder().streamId(streamId)
                .ctx(context).channel(channel).aeron(aeron)
                .build()) {
            publisher.publish(arrGet);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
