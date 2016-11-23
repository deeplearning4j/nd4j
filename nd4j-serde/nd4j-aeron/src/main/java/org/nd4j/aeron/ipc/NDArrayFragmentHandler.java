package org.nd4j.aeron.ipc;

import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import lombok.extern.slf4j.Slf4j;
import org.agrona.DirectBuffer;
import org.nd4j.aeron.ipc.chunk.ChunkAccumulator;
import org.nd4j.aeron.ipc.chunk.InMemoryChunkAccumulator;
import org.nd4j.aeron.ipc.chunk.NDArrayMessageChunk;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;


/**
 * NDArray fragment handler
 * for listening to an aeron queue
 *
 * @author Adam Gibson
 */
@Slf4j
public class NDArrayFragmentHandler implements FragmentHandler {
    private NDArrayCallback ndArrayCallback;
    private ChunkAccumulator chunkAccumulator = new InMemoryChunkAccumulator();

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
    public   void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        ByteBuffer byteBuffer = buffer.byteBuffer();
        boolean byteArrayInput = false;
        if(byteBuffer == null) {
            byteArrayInput = true;
            byte[] destination = new byte[length];
            ByteBuffer wrap = ByteBuffer.wrap(buffer.byteArray());
            wrap.get(destination,offset,length);
            byteBuffer = ByteBuffer.wrap(destination).order(ByteOrder.nativeOrder());
        }


        //only applicable for direct buffers where we don't wrap the array
        if(!byteArrayInput) {
            byteBuffer.position(offset);
            byteBuffer.order(ByteOrder.nativeOrder());
        }

        NDArrayMessage.MessageType messageType = NDArrayMessage.MessageType.values()[byteBuffer.getInt()];

        if(messageType == NDArrayMessage.MessageType.CHUNKED) {
            NDArrayMessageChunk chunk = NDArrayMessageChunk.fromBuffer(byteBuffer,messageType);
            if(chunk.getNumChunks() < 1)
                throw new IllegalStateException("Found invalid number of chunks " + chunk.getNumChunks() + " on chunk index " + chunk.getChunkIndex());
            chunkAccumulator.accumulateChunk(chunk);
            log.info("Number of chunks " + chunk.getNumChunks() + " and number of chunks " + chunk.getNumChunks() + " for id " + chunk.getId() + " is " + chunkAccumulator.numChunksSoFar(chunk.getId()));

            if(chunkAccumulator.allPresent(chunk.getId())) {
                NDArrayMessage message = chunkAccumulator.reassemble(chunk.getId());
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
        else {
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
}
