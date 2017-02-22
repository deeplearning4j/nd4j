package org.nd4j.jita.allocator.pointers.cuda;

import lombok.Getter;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
public class cudaEvent_t extends CudaPointer {

    private AtomicBoolean destroyed = new AtomicBoolean(false);

    @Getter
    @Setter
    private long clock;

    @Getter
    @Setter
    private int laneId;

    @Getter
    @Setter
    private int deviceId;

    public cudaEvent_t(Pointer pointer) {
        super(pointer);
    }

    public boolean isDestroyed() {
        return destroyed.get();
    }

    public void markDestoryed() {
        destroyed.set(true);
    }

    public void destroy() {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().destroyEvent(this);
            markDestoryed();
        }
    }

    public void synchronize() {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().eventSynchronize(this);
        }
    }

    public void register(cudaStream_t stream) {
        if (!isDestroyed()) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().registerEvent(this, stream);
        }
    }
}
