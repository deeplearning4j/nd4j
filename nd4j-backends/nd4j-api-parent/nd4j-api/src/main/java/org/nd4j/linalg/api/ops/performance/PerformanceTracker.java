package org.nd4j.linalg.api.ops.performance;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ops.performance.primitives.AveragedHolder;
import org.nd4j.linalg.factory.Nd4j;

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

    private Map<Integer, AveragedHolder> bandwidth = new HashMap<>();
    private Map<Integer, AveragedHolder> operations = new HashMap<>();

    private PerformanceTracker() {
        // we put in initial holders, one per device
        val nd = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int e = 0; e < nd; e++) {
            bandwidth.put(e, new AveragedHolder());
            operations.put(e, new AveragedHolder());
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
        // we calculate bytes per microsecond now
        val bw = (long) (numberOfBytes / (timeSpentNanos / (double) 1000.0));

        bandwidth.get(deviceId).addValue(bw);

        return bw;
    }
}
