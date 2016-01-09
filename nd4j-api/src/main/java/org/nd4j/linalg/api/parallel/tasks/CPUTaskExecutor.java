package org.nd4j.linalg.api.parallel.tasks;

import org.nd4j.linalg.executors.ExecutorServiceProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.*;

public class CPUTaskExecutor implements TaskExecutor {

    private static CPUTaskExecutor instance;
    private static final int nThreads;
    static {
        int defaultThreads = Runtime.getRuntime().availableProcessors();
        boolean enabled = Boolean.parseBoolean(System.getProperty(ExecutorServiceProvider.ENABLED,"true"));
        if(!enabled) nThreads = 1;
        else nThreads = Integer.parseInt(System.getProperty(ExecutorServiceProvider.EXEC_THREADS,String.valueOf(defaultThreads)));
        instance = new CPUTaskExecutor(nThreads);
    }

    public static TaskExecutor getInstance(){
        return instance;
    }

    private final Thread[] workerThreads;
    private final WorkerRunnable[] workers;

    public CPUTaskExecutor(int nThreads){

        workerThreads = new Thread[nThreads];
        workers = new WorkerRunnable[nThreads];
        for( int i=0; i<nThreads; i++ ){
            workers[i] = new WorkerRunnable();
            workerThreads[i] = new Thread(workers[i]);
            workerThreads[i].setDaemon(true);
            workerThreads[i].setName("CPUTaskExecutor-" + i);
            workerThreads[i].start();
        }
    }

    @Override
    public <V> Future<V> executeAsync(Task<V> task) {
        for(int i=0; i<workers.length; i++ ){
            workers[i].tasks.add(task);
        }
        return (Future<V>)task;
    }

    private static class WorkerRunnable implements Runnable {
        private Logger log = LoggerFactory.getLogger(WorkerRunnable.class);

        private BlockingQueue<Task<?>> tasks = new LinkedTransferQueue<>();


        @Override
        public void run() {
            while(true){
                Task<?> task;
                try{
                    task = tasks.take();
                }catch(InterruptedException e){
                    log.error("Interrupted exception: Shutting down. {}",Thread.currentThread().getId(),e);
                    throw new RuntimeException(e);
                }
                try {
                    task.call();
                } catch(Exception e){
                    log.warn("Exception thrown by task {} on threadId={}",task,Thread.currentThread().getName(),e);
                    e.printStackTrace();
                    ((Future<?>)task).cancel(true);
                }
            }
        }
    }

}
