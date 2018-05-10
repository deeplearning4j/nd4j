package org.nd4j.linalg.api.ops.performance;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ops.performance.primitives.AveragingTransactionsHolder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;

import java.util.HashMap;
import java.util.Map;

/**
 * This class provides routines for performance tracking and holder for corresponding results
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class PerformanceTracker {
    private static final PerformanceTracker INSTANCE = new PerformanceTracker();

    private Map<Integer, AveragingTransactionsHolder> bandwidth = new HashMap<>();
    private Map<Integer, AveragingTransactionsHolder> operations = new HashMap<>();

    private PerformanceTracker() {
        // we put in initial holders, one per device
        val nd = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int e = 0; e < nd; e++) {
            bandwidth.put(e, new AveragingTransactionsHolder());
            operations.put(e, new AveragingTransactionsHolder());
        }
    }

    public static PerformanceTracker getInstance() {
        return INSTANCE;
    }

    /**
     * This method stores bandwidth used for given transaction.
     *
     * PLEASE NOTE: Bandwidth is stored in per millisecond value.
     *
     * @param deviceId device used for this transaction
     * @param timeSpent time spent on this transaction in nanoseconds
     * @param numberOfBytes number of bytes
     */
    public long addMemoryTransaction(int deviceId, long timeSpentNanos, long numberOfBytes) {
        // default is H2H transaction
        return addMemoryTransaction(deviceId, timeSpentNanos, numberOfBytes, MemcpyDirection.HOST_TO_HOST);
    }

    /**
     * This method stores bandwidth used for given transaction.
     *
     * PLEASE NOTE: Bandwidth is stored in per millisecond value.
     *
     * @param deviceId device used for this transaction
     * @param timeSpent time spent on this transaction in nanoseconds
     * @param numberOfBytes number of bytes
     * @param direction direction for the given memory transaction
     */
    public long addMemoryTransaction(int deviceId, long timeSpentNanos, long numberOfBytes, @NonNull MemcpyDirection direction) {
        // we calculate bytes per microsecond now
        val bw = (long) (numberOfBytes / (timeSpentNanos / (double) 1000.0));

        bandwidth.get(deviceId).addValue(direction, bw);

        return bw;
    }

    public void clear() {
        for (val k: bandwidth.keySet())
            bandwidth.get(k).clear();
    }
}
