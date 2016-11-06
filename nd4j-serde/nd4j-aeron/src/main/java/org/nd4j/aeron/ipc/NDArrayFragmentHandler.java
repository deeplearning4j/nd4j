package org.nd4j.aeron.ipc;

import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * NDArray fragment handler
 * for listening to an aeron queue
 *
 * @author Adam Gibson
 */
public class NDArrayFragmentHandler implements FragmentHandler {
    private NDArrayCallback ndArrayCallback;


    public NDArrayFragmentHandler(NDArrayCallback ndArrayCallback) {
        this.ndArrayCallback = ndArrayCallback;
    }

    /**
     * Callback for handling
     * fragments of data being read from a log.
     *
     * @param buffer containing the data.
     * @param offset at which the data begins.
     * @param length of the data in bytes.
     * @param header representing the meta data for the data.
     */
    @Override
    public void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        buffer = new UnsafeBuffer(buffer,offset,length);
        NDArrayMessage message = NDArrayMessage.fromBuffer(buffer,offset);
        INDArray arr = message.getArr();
        //of note for ndarrays
        int[] dimensions = message.getDimensions();
        boolean whole = dimensions.length == 1 && dimensions[0] == -1;

        if(!whole)
            ndArrayCallback.onNDArrayPartial(arr,message.getIndex(),dimensions);
        else
            ndArrayCallback.onNDArray(arr);

    }
}
