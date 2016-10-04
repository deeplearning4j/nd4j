package org.nd4j.parameterserver.client;

import io.aeron.Aeron;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.aeron.ipc.*;
import org.nd4j.aeron.ipc.response.HostPortPublisher;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Parameter server client for
 * publishing and
 * retrieving ndarrays
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
@Builder
public class ParameterServerClient implements NDArrayCallback {
    //the url to send ndarrays to
    private String ndarraySendUrl;
    //the url to retrieve ndarrays from
    private String ndarrayRetrieveUrl;
    private Aeron.Context ctx;
    private AeronNDArraySubscriber subscriber;
    //host to listen on for the subscriber
    private String subscriberHost;
    //port to listen on for the subscriber
    private int subscriberPort;
    //the stream to listen on for the subscriber
    private int subscriberStream = 11;
    //the "current" ndarray
    private INDArray arr;

    /**
     * Push an ndarray to the specified
     * ndarray send url in the form of:
     * host;port:stream
     * where stream is the stream for connecting
     * to a listening aeron server
     * @param arr the array to send
     */
    public void pushNDArray(INDArray arr) {
        String[] split = ndarraySendUrl.split(":");
        int port = Integer.parseInt(split[1]);
        int streamToPublish = Integer.parseInt(split[2]);
        String channel = AeronUtil.aeronChannel(split[0],port);
        AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder()
                .streamId(streamToPublish)
                .ctx(ctx).channel(channel)
                .build();
        try {
            publisher.publish(arr);
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            publisher.close();
        } catch (Exception e) {

        }

    }


    /**
     *
     * @return
     */
    public INDArray getArray() {
        //start a subscriber that can send us ndarrays
        if(subscriber == null) {
            subscriber = AeronNDArraySubscriber.startSubscriber(
                    ctx,
                    subscriberHost,
                    subscriberPort,
                    this,
                    subscriberStream);
        }

        if(arr == null) {
            //note here that this is the "master url"
            String[] split = ndarrayRetrieveUrl.split(":");
            //The response daemon is always the master daemon's port + 1
            //A "master daemon" is one that holds both the
            //parameter averaging daemon AND the response daemon for being able to send
            //the "current state ndarray"
            int port = Integer.parseInt(split[1]);
            int streamToPublish = Integer.parseInt(split[2]);
            //the channel here is the master node host with the port + 1
            //pointing at the response node where we can request ndarrays to be sent to
            //the listening daemon
            String channel = AeronUtil.aeronChannel(split[0],port);
            //publish the address of our subscriber
            //note here that we send the ndarray send url, because the
            //master also hosts
            HostPortPublisher hostPortPublisher = HostPortPublisher
                    .builder().channel(channel).ctx(ctx)
                    //note here that we send our subscriber's listening information
                    .streamId(streamToPublish)
                    .uriToSend(AeronConnectionInformation.of(subscriberHost,subscriberPort,subscriberStream).toString())
                    .build();
            try {
                hostPortPublisher.send();
            } catch (Exception e) {
                e.printStackTrace();
            }

            try {
                hostPortPublisher.close();
            } catch (Exception e) {

            }

            while(arr == null) {
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        else {
            INDArray currentArr = this.arr;
            this.arr = null;
            return currentArr;
        }


        return arr;
    }

    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public void onNDArray(INDArray arr) {
        this.arr = arr;
    }
}
