package org.nd4j.linalg.api.ops.performance.primitives;

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@NoArgsConstructor
public class AveragedHolder {
    private final List<Long> storage = new ArrayList<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    public void addValue(Long value) {
        try {
            lock.writeLock().lock();

            storage.add(value);
        } finally {
            lock.writeLock().unlock();
        }
    }

    public Long getAverageValue() {
        try {
            Long r = 0L;
            lock.readLock().lock();

            for (val v : storage)
                r += v;

            return r / storage.size();
        } finally {
            lock.readLock().unlock();
        }
    }
}
