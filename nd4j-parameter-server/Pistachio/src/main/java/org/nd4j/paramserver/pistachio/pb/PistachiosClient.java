/*
 * Copyright 2014 Yahoo! Inc. Licensed under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
 * agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */


package org.nd4j.paramserver.pistachio.pb;

import org.nd4j.paramserver.pistachio.pb.kafka.KeyValue;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;

import org.apache.commons.configuration.Configuration;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

import org.slf4j.Logger;
import org.apache.zookeeper.ZooKeeper;
import org.slf4j.LoggerFactory;

import com.google.api.client.util.ExponentialBackOff;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Meter;
import com.codahale.metrics.Timer;
import com.codahale.metrics.JmxReporter;
import com.google.api.client.util.BackOff;
import org.nd4j.paramserver.pistachio.pb.customization.LookupCallbackRegistry;
import org.nd4j.paramserver.pistachio.pb.customization.ProcessorRegistry;
import org.nd4j.paramserver.pistachio.pb.exception.*;
import org.nd4j.paramserver.pistachio.pb.network.netty.NettyPistachioClient;

import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.KeeperException.NodeExistsException;

import org.nd4j.paramserver.pistachio.pb.customization.StoreCallbackRegistry;

/**
 * Main Pistachio Client Class
 * <ul>
 * <li>To use the Client, new an instance and call the functions
 * </ul>
 * <p>
 *
 * @author      Gavin Li
 * @version     0.1.0.0
 * @since       1.0
 */
public class PistachiosClient {
    private Configuration conf = ConfigurationManager.getConfiguration();
    private static Logger logger = LoggerFactory.getLogger(PistachiosClient.class);
    final static MetricRegistry metrics = new MetricRegistry();
    final static JmxReporter reporter = JmxReporter.forRegistry(metrics).inDomain("pistachio.client.metrics").build();

    private final static Meter lookupFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "lookupFailureRequests"));
    private final static Meter storeFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "storeFailureRequests"));
    private final static Meter processFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "processFailureRequests"));
    private final static Meter multiLookupFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "multiLookupFailureRequests"));
    private final static Meter multiLookupAsyncFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "multiLookupAsyncFailureRequests"));
    private final static Meter multiProcessAsyncFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "multiProcessAsyncFailureRequests"));
    private final static Meter storeAsyncFailureRequests = metrics.meter(MetricRegistry.name(PistachiosServer.class, "storeAsyncFailureRequests"));

    private final static Timer lookupTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "lookupTimer"));
    private final static Timer storeTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "storeTimer"));
    private final static Timer processTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "processTimer"));
    private final static Timer multiLookupTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "multiLookupTimer"));
    private final static Timer multiLookupAsyncTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "multiLookupAsyncTimer"));
    private final static Timer multiProcessAsyncTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "multiLookupAsyncTimer"));
    private final static Timer storeAsyncTimer = metrics.timer(MetricRegistry.name(PistachiosServer.class, "storeAsyncTimer"));

    private PistachiosClientImpl clientImpl = new NettyPistachioClient();

    static private class DummyWatcher implements Watcher {
        public void    process(WatchedEvent event) {
        }
    }



    static {
        reporter.start();
        ZooKeeper zk =  null;
        try {
            zk = new ZooKeeper(ConfigurationManager.getConfiguration().getString("Pistachio.ZooKeeper.Server"),40000,new DummyWatcher());
            setZkRegistryData(ConfigurationManager.getConfiguration().getString("Pistachio.Processor.JarPath"),
                              ConfigurationManager.getConfiguration().getString("Pistachio.Processor.ClassName"),
                              ProcessorRegistry.processorRegistryPath,
                              zk);
            setZkRegistryData(ConfigurationManager.getConfiguration().getString("Pistachio.StoreCallback.JarPath"),
                              ConfigurationManager.getConfiguration().getString("Pistachio.StoreCallback.ClassName"),
                              StoreCallbackRegistry.storeCallbackPath,
                              zk);
            setZkRegistryData(ConfigurationManager.getConfiguration().getString("Pistachio.LookupCallback.JarPath"),
                    ConfigurationManager.getConfiguration().getString("Pistachio.LookupCallback.ClassName"),
                    LookupCallbackRegistry.lookupCallbackPath,
                    zk);
        } catch (Exception e) {
            logger.info("error updating zk", e);
        } finally {
            try {
            if (zk != null)
                zk.close();
            } catch (Exception e) {
            }
        }
    }

    private static void setZkRegistryData(String jarPath, String className, String zkPath, ZooKeeper zk){
        try {
            if (jarPath != null && className != null) {
                String value = (jarPath + ";" + className);
                logger.info("updating {} to {}", zkPath, value);
                String currentPath = "";
                for (String path : zkPath.split("/")) {
                    if (path.isEmpty())
                        continue;
                    currentPath = currentPath + "/" + path;
                    try {
                        logger.info("creating {}", currentPath);
                        zk.create(currentPath, value.getBytes(), org.apache.zookeeper.ZooDefs.Ids.OPEN_ACL_UNSAFE , org.apache.zookeeper.CreateMode.PERSISTENT);
                    } catch (NodeExistsException e0) {
                    } catch (Exception e) {
                        logger.info("error creating path {}", path, e);
                        break;
                    }
                }
                zk.setData(zkPath, value.getBytes(), -1);
                logger.info("updated {} to {}", zkPath, value);
            }
        } catch (Exception e) {
            logger.info("error updating zk", e);
        }
    }
    private int initialIntervalMillis = conf.getInt("Pistachio.AutoRetry.BackOff.InitialIntervalMillis", 100);
    private int maxElapsedTimeMillis = conf.getInt("Pistachio.AutoRetry.BackOff.MaxElapsedTimeMillis", 100 * 1000);
    private int maxIntervalMillis = conf.getInt("Pistachio.AutoRetry.BackOff.MaxIntervalMillis", 5000);
    private Boolean noMasterAutoRetry = conf.getBoolean("Pistachio.NoMasterAutoRetry", true);
    private Boolean connectionBrokenAutoRetry = conf.getBoolean("Pistachio.ConnectionBrokenAutoRetry", true);

    private class RetryWaiter {
        long backOffMillis =  0;
        BackOff backoff;
        Meter failureMeter;
        public RetryWaiter(Meter failureMeter0) {
            failureMeter = failureMeter0;

            backoff = (new ExponentialBackOff.Builder()).setInitialIntervalMillis(initialIntervalMillis)
                .setMaxElapsedTimeMillis(maxElapsedTimeMillis)
                .setMaxIntervalMillis(maxIntervalMillis)
                .build();
        }
        void waitBeforeRetry(Exception me) throws Exception {
            if (me instanceof MasterNotFoundException && !noMasterAutoRetry) {
                failureMeter.mark();
                throw me;
            }

            if (me instanceof ConnectionBrokenException && !connectionBrokenAutoRetry) {
                failureMeter.mark();
                throw me;
            }

            try{
                backOffMillis = backoff.nextBackOffMillis();
                if (backOffMillis == BackOff.STOP) {
                    failureMeter.mark();
                    throw me;
                }
                logger.debug("no master found, auto retry after sleeping {} ms", backOffMillis);
                Thread.sleep(backOffMillis);
            }catch(Exception e) {
            }
        }
    }

    public PistachiosClient() throws Exception {

    }

    /**
     * To lookup the value of an id. Given the id return the value as a byte array.
     *
     * @param id        id to look up as byte[].
     * @return          <code>byte array</code> return in byte array, null if key not found
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public byte[] lookup(byte[] id) throws Exception {
        return lookup(id, false);
    }

    /**
     * To lookup the value of an id. Given the id return the value as a byte array. One can also register callback
     * on the server side upon lookup, see {@link org.nd4j.paramserver.pistachio.pb.customization.LookupCallback}
     *
     * @param id        id to look up as byte[].
     * @param callback  whether need callback on server side or not
     * @return          <code>byte array</code> return in byte array, null if key not found
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public byte[] lookup(byte[] id, boolean callback) throws Exception {

        final Timer.Context context = lookupTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(lookupFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.lookup(id, callback);
                } catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }

            }

        } finally {
            context.stop();
        }

    }

    /**
     * To lookup a list of ids. Given the id list return the values.
     *
     * @param ids       id to look up as list of byte[].
     * @return          <code>Map<byte[], byte[]></code> return in a map of values for each corresponding ids
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Map<byte[], byte[]> multiLookUp(List<byte[]> ids) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        return multiLookUp(ids, false);
    }

    /**
     * To lookup a list of ids. Given the id list return the values. One can also register callback
     * on the server side upon lookup, see {@link org.nd4j.paramserver.pistachio.pb.customization.LookupCallback}
     *
     * @param ids       id to look up as list of byte[].
     * @param callback  whether need callback on server side or not
     * @return          <code>Map<byte[], byte[]></code> return in a map of values for each corresponding ids
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Map<byte[], byte[]> multiLookUp(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        final Timer.Context context = multiLookupTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(multiLookupFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.multiLookup(ids, callback);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }

        } finally {
            context.stop();
        }
    }

    /**
     * To lookup a list of ids asynchronously. Given the id list return the futures to get the values.
     *
     * @param ids       id to look up as list of byte[].
     * @return          <code>Map<byte[], Future<byte[]>></code> return in a map of futre of value for each corresponding ids
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Future<Map<byte[], byte[]>> multiLookUpAsync(List<byte[]> ids) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        return multiLookUpAsync(ids, false);
    }

    /**
     * To lookup a list of ids asynchronously. Given the id list return the futures to get the values. One can also register callback
     * on the server side upon lookup, see {@link org.nd4j.paramserver.pistachio.pb.customization.LookupCallback}
     *
     * @param ids       id to look up as list of byte[].
     * @param callback  whether need callback on the same side or not
     * @return          <code>Map<byte[], Future<byte[]>></code> return in a map of futre of value for each corresponding ids
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Future<Map<byte[], byte[]>> multiLookUpAsync(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        final Timer.Context context = multiLookupAsyncTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(multiLookupAsyncFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.multiLookupAsync(ids, callback);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }
        } finally {
            context.stop();
        }
    }

    /**
     * To store the key value without calling callback.
     *
     * @param id        id to store as byte[].
     * @param value     value to store as byte array
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     * @return          <code>boolean</code> succeeded or not
     */
    public boolean store(byte[] id, byte[] value)  throws MasterNotFoundException, ConnectionBrokenException, Exception{
        return store(id, value, true);
    }

    /**
     * To store the key value.
     *
     * @param id        id to store as byte[].
     * @param value     value to store as byte array
     * @param callback  does the registered callback need to be triggered
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     * @return          <code>boolean</code> succeeded or not
     */
    public boolean store(byte[] id, byte[] value, boolean callback)  throws  Exception {
        final Timer.Context context = storeTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(storeFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.store(id, value, callback);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }
        } finally {
            context.stop();
        }
    }

    /**
     * To close all the resource gracefully
     */
    public void close() {
        clientImpl.close();
    }

    /**
     * To process a batch of events
     *
     * @param id        id to store as byte[].
     * @param events    list of events as byte []
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     * @return          <code>boolean</code> succeeded or not
     */
    public boolean processBatch(byte[] id, List<byte[]> events) throws MasterNotFoundException, ConnectionBrokenException, Exception{
        final Timer.Context context = processTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(processFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.processBatch(id, events);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }
        } finally {
            context.stop();
        }
    }

    /**
     * To process a batch of events
     *
     * @param events    Mapping keeps id (byte[]) -> event (byte[])
     * @return          <code>List<Future<Boolean>></code> succeeded or not
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Future<Map<byte[], Boolean>> multiProcessAsync(Map<byte[], byte[]> events)  throws MasterNotFoundException, ConnectionBrokenException, Exception{
        final Timer.Context context = multiProcessAsyncTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(multiProcessAsyncFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.multiProcessAsync(events);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }
        } finally {
            context.stop();
        }
    }

    /**
     * To store the key value.
     *
     * @param id        id to store as byte[].
     * @param value     value to store as byte array
     * @return          <code>boolean</code> succeeded or not
     * @exception       MasterNotFoundException when fail because no master found
     * @exception       ConnectionBrokenException when fail because connection is broken in the middle
     * @exception       Exception other errors indicating failure
     */
    public Future<Boolean> storeAsync(byte[] id, byte[] value) throws MasterNotFoundException, ConnectionBrokenException, Exception{
        final Timer.Context context = storeAsyncTimer.time();
        RetryWaiter retryWaiter = new RetryWaiter(storeAsyncFailureRequests);

        try {
            while (true) {
                try {
                    return clientImpl.storeAsync(id, value);
                }catch (Exception e) {
                    retryWaiter.waitBeforeRetry(e);
                }
            }
        } finally {
            context.stop();
        }
    }

   /**
    * To delete a record belong to a key.
    *
    * @param key        key to store as byte[].
    * @return          <code>boolean</code> succeeded or not
    * @exception       MasterNotFoundException when fail because no master found
    * @exception       ConnectionBrokenException when fail because connection is broken in the middle
    * @exception       Exception other errors indicating failure
    */
    public boolean delete(byte[] key) throws MasterNotFoundException,
            ConnectionBrokenException {
        return clientImpl.delete(key);
    }

   /**
    * Get iterator of a partition
    *
    * @param partition partition to iterate
    * @return          <code>PistachioIterator</code> return a pistachioIterator
    */
    public PistachioIterator iterator(long partition) throws MasterNotFoundException,
            ConnectionBrokenException {
        return clientImpl.iterator(partition);
    }

    public static void main(String [] args) {
        PistachiosClient client = null;
        try {
            client = new PistachiosClient();
        } catch (Exception e) {
            logger.info("error creating clietn", e);
            if (client != null)
                client.close();
            return;
        }

        try {

            String id = "";
            boolean store = false;
            String value="" ;
            if (args.length == 2 && args[0].equals("lookup") ) {
                id = args[1];
                System.out.println("client.lookup(" + id + ")" + new String(client.lookup(id.getBytes(), true)));
            } else if (args.length == 3 && args[0].equals("store") ) {
                id = args[1];
                store = true;
                value = args[2];
                client.store(id.getBytes(), value.getBytes(), true);
            } else if (args.length == 3 && args[0].equals("processbatch") ) {
                    id = (args[1]);
                store = true;
                value = args[2];
                List list = new java.util.ArrayList();
                list.add(value.getBytes());
                client.processBatch(id.getBytes(), list);
              } else if (args.length == 2 && args[0].equals("delete")) {
                  System.out.println("you are deleting "+ id);
                  try {
                      id = args[1];
                  } catch (Exception e) {
                  }
                  client.delete(id.getBytes());
              } else if (args.length == 2 && args[0].equals("iterate")) {
                  String partition = args[1];
                  PistachioIterator iterator = client.iterator(Long.parseLong(partition));
                  KeyValue keyValue = iterator.getNext();
                  while (keyValue != null) {
                      System.out.println("key :" + new String(keyValue.key));
                      System.out.println("value :" + new String(keyValue.value));
                      keyValue = iterator.getNext();
                  }
                  System.out.println("you are iterate partition " + id);
              } else if (args.length == 3 && args[0].equals("jump")) {
                  String partition = args[1];
                  PistachioIterator iterator = client.iterator(Long.parseLong(partition));
                  iterator.jump(args[2].getBytes());
                  KeyValue keyValue = iterator.getNext();
                  while (keyValue != null) {
                      System.out.println("key :" + new String(keyValue.key));
                      System.out.println("value :" + new String(keyValue.value));
                      keyValue = iterator.getNext();
                  }
                  System.out.println("you are iterate partition " + id);
              } else if (args.length == 2 && args[0].equalsIgnoreCase("multilookup")) {
                  String[] tokens = args[1].split(",");
                  List<byte[]> ids = new ArrayList<>(tokens.length);
                  for (String token: tokens) {
                      ids.add(token.getBytes());
                  }
                  Map<byte[], byte[]> ret = client.multiLookUp(ids, true);
                  for (Map.Entry<byte[], byte[]> entry: ret.entrySet()) {
                        System.out.println("client.lookup(" + DefaultDataInterpreter.getDataInterpreter().interpretId(entry.getKey()) + ")"
                                + DefaultDataInterpreter.getDataInterpreter().interpretData(entry.getValue()));
                  }
              } else if (args.length == 3 && args[0].equalsIgnoreCase("multiprocess")) {
                  String[] tokens = args[1].split(",");
                  value = args[2];
                  Map<byte[], byte[]> events = new HashMap<>(tokens.length);
                  for (String token: tokens) {
                      events.put(token.getBytes(), value.getBytes());
                  }
                  List<byte[]> list = new ArrayList<>();
                  list.add(value.getBytes());

                  Map<byte[], Boolean> ret = client.multiProcessAsync(events).get();
                  for (Map.Entry<byte[], Boolean> entry: ret.entrySet()) {
                        System.out.println("client.lookup(" + DefaultDataInterpreter.getDataInterpreter().interpretId(entry.getKey()) + ") "
                                + (entry.getValue() ? "Succeed" : "Failure"));
                  }
              } else {
                  System.out.println("USAGE: xxxx lookup id or xxxx store id value");
                  System.exit(0);
              }
          } catch (Exception e) {
            System.out.println("error: "+ e);
        } finally {
            client.close();
        }




    /*
    if (args.length != 1) {
      System.out.println("Please enter 'simple' or 'secure'");
      System.exit(0);
    }
    */

  }

}
