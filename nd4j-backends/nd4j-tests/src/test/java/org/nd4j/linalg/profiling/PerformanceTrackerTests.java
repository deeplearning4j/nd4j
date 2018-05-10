package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.performance.primitives.AveragedHolder;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class PerformanceTrackerTests extends BaseNd4jTest {
    public PerformanceTrackerTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testAveragedHolder_1() {
        val holder = new AveragedHolder();

        holder.addValue(50L);
        holder.addValue(150L);

        assertEquals(100L, holder.getAverageValue().longValue());
    }

    @Test
    public void testAveragedHolder_2() {
        val holder = new AveragedHolder();

        holder.addValue(50L);
        holder.addValue(150L);
        holder.addValue(100L);

        assertEquals(100L, holder.getAverageValue().longValue());
    }

    @Test
    public void testPerformanceTracker_1() {
        val perf = PerformanceTracker.getInstance();

        // 100 nanoseconds spent for 5000 bytes. result should be around 50000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 100, 5000);
        assertEquals(50000, res);
    }

    @Test
    public void testPerformanceTracker_2() {
        val perf = PerformanceTracker.getInstance();

        // 10 nanoseconds spent for 5000 bytes. result should be around 500000 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10, 5000);
        assertEquals(500000, res);
    }

    @Test
    public void testPerformanceTracker_3() {
        val perf = PerformanceTracker.getInstance();

        // 10000 nanoseconds spent for 5000 bytes. result should be around 500 bytes per microsecond
        val res = perf.addMemoryTransaction(0, 10000, 5000);
        assertEquals(500, res);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
