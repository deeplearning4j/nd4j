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
package org.nd4j.paramserver.pistachio.pb.network.netty;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.google.common.base.Function;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.protobuf.ByteString;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import org.apache.commons.configuration.Configuration;
import org.nd4j.paramserver.pistachio.pb.*;
import org.nd4j.paramserver.pistachio.pb.exception.ConnectionBrokenException;
import org.nd4j.paramserver.pistachio.pb.exception.MasterNotFoundException;
import org.nd4j.paramserver.pistachio.pb.exception.RequestTimeoutException;
import org.nd4j.paramserver.pistachio.pb.helix.HelixPartitionSpectator;
import org.nd4j.paramserver.pistachio.pb.kafka.KeyValue;
import org.nd4j.paramserver.pistachio.pb.network.netty.NettyPistachioProtocol.Request;
import org.nd4j.paramserver.pistachio.pb.network.netty.NettyPistachioProtocol.RequestType;
import org.nd4j.paramserver.pistachio.pb.network.netty.NettyPistachioProtocol.Response;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.nd4j.paramserver.pistachio.pb.util.NativeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetAddress;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Sends a list of continent/city pairs to a {@link NettyPistachioServer} to
 * get the local times of the specified cities.
 */
public final class NettyPistachioClient implements PistachiosClientImpl{
    private static Logger logger = LoggerFactory.getLogger(NettyPistachioClient.class);

    static final boolean SSL = System.getProperty("ssl") != null;
    static final String HOST = System.getProperty("host", "127.0.0.1");
    //    static final int PORT = Integer.parseInt(System.getProperty("port", "8463"));
    static final int PORT = ConfigurationManager.getConfiguration().getInt("Network.Netty.Port", 9091);

    private NettyPistachioClientHandler handler;
    private List<Channel> channelList = new ArrayList<>();

    static private ConcurrentHashMap<String, Channel> hostChannelMap = new ConcurrentHashMap<>();
    private EventLoopGroup eventLoopGroup = new NioEventLoopGroup();
    static private HelixPartitionSpectator helixPartitionSpectator = null;
    static final String ZOOKEEPER_SERVER = "Pistachio.ZooKeeper.Server";
    static final String PROFILE_HELIX_INSTANCE_ID = "Profile.Helix.InstanceId";
    public final static String RESOURCE = "PistachiosResource";
    public final static String CLUSTER = "PistachiosCluster";
    public final static String STATE = "MASTER";
    private Configuration conf = ConfigurationManager.getConfiguration();
    private Partitioner partitioner = new DefaultPartitioner();
    private String localHostAddress;

    public NettyPistachioClient() throws Exception{
        localHostAddress = InetAddress.getLocalHost().getHostAddress();

        if (helixPartitionSpectator == null) {
            synchronized(this) {
                if (helixPartitionSpectator == null) {
                    try {
                        helixPartitionSpectator = HelixPartitionSpectator.getInstance(
                                conf.getString(ZOOKEEPER_SERVER), // zkAddr
                                CLUSTER,
                                NativeUtils.getHostname()//InetAddress.getLocalHost().getHostName() //conf.getString(PROFILE_HELIX_INSTANCE_ID) // instanceName
                        );
                    } catch(Exception e) {
                        logger.error("Error init HelixPartitionSpectator, are zookeeper and helix installed and configured correctly?", e);
                        throw e;
                    }
                }
            }
        }
    }

    private boolean isLocalCall(long partition) {
        boolean directLocalCall = false;
        String ip = helixPartitionSpectator.getOneInstanceForPartition(RESOURCE, (int)partition, STATE);

        try {
            logger.debug("islocal {}, ip {}, localAddr {}, serving {}",
                    partition, ip, localHostAddress, PistachiosServer.servingAsServer());

            if (ip == null || (ip != null) && ip.equals(localHostAddress) && PistachiosServer.servingAsServer()) {
                directLocalCall = true;
            }
        } catch (Exception e) {
            logger.info("error:", e);
        }

        return directLocalCall;
    }

    private NettyPistachioClientHandler getHandler(byte[] id)  throws  Exception {

        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
        logger.debug("get partition {} for id {}", partition, id);
        return getHandlerForPartition(partition);
    }

    private NettyPistachioClientHandler getHandlerForPartition(long partition)  throws  Exception {

        String ip = helixPartitionSpectator.getOneInstanceForPartition(RESOURCE, (int)partition, STATE);
        logger.debug("ip {} found for partition {}", ip, partition);

        if (ip == null) {
            logger.info("partition master not found for partition{}", partition);
            throw new MasterNotFoundException("master partition not found");
        }

        if (hostChannelMap.containsKey(ip) && !hostChannelMap.get(ip).isActive()) {
            synchronized(hostChannelMap) {
                if (hostChannelMap.containsKey(ip) && !hostChannelMap.get(ip).isActive()) {
                    //TODO: add JMX metrics for disconnections and reconn
                    logger.info("ip {} , partition {} connection not active any more reconnecting", ip, partition);
                    hostChannelMap.get(ip).close();
                    hostChannelMap.remove(ip);
                }
            }
        }
        NettyPistachioClientHandler handler = null;

        if (!hostChannelMap.containsKey(ip)) {
            synchronized(hostChannelMap) {
                if (!hostChannelMap.containsKey(ip)) {
                    try {
                        // Configure SSL.
                        final SslContext sslCtx;
                        if (SSL) {
                            sslCtx = SslContext.newClientContext(InsecureTrustManagerFactory.INSTANCE);
                        } else {
                            sslCtx = null;
                        }

                        Bootstrap b = new Bootstrap();
                        b.group(eventLoopGroup)
                                .channel(NioSocketChannel.class)
                                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, conf.getInt("Network.Netty.ClientConnectionTimeoutMillis", 100000))
                                .option(ChannelOption.SO_SNDBUF, 1048576)
                                .option(ChannelOption.SO_RCVBUF, 1048576)
                                .handler(new NettyPistachioClientInitializer(sslCtx));

                        // Make a new connection.
                        Channel ch = b.connect(ip, PORT).sync().channel();
                        channelList.add(ch);

                        // Get the handler instance to initiate the request.
                        handler = ch.pipeline().get(NettyPistachioClientHandler.class);
                        handler.setIp(ip);
                        hostChannelMap.put(ip, ch);

                    } catch (Throwable e) {
                        logger.info("error constructor ", e);
                        throw new MasterNotFoundException(e.getMessage());
                    }
                }
            }
        }
        handler = hostChannelMap.get(ip).pipeline().get(NettyPistachioClientHandler.class);
        return handler;
    }

    @Override
    public boolean delete(byte[] id) throws MasterNotFoundException, ConnectionBrokenException{
        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
        if (isLocalCall(partition)) {
            return PistachiosServer.handler.delete(id, partition);
        }
        NettyPistachioClientHandler handler = null;
        try {
            handler = getHandler(id);
        } catch (Exception e) {
            logger.info("error getting handler", e);
            return false;
        }
        if (handler == null) {
            logger.debug("fail to look up {} because of empty handler", id);
            return false;
        }
        Request.Builder builder = Request.newBuilder();
        Response res = handler.sendRequest(builder.setId(ByteString.copyFrom(id)).setType(RequestType.DELETE).setPartition(partition));

        if (res == null) {
            logger.debug("fail");
            return false;
        }
        return res.getSucceeded();
    }

    public void close() {
        for (Channel channel: channelList) {
            channel.close();
        }
        eventLoopGroup.shutdownGracefully();
        logger.info("close called");
    }

    public byte[] lookup(byte[] id, boolean callback)  throws  Exception {
        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition("PistachiosResource"));
        if (isLocalCall(partition)) {
            return PistachiosServer.handler.lookup(id, partition, callback);
        }
        NettyPistachioClientHandler handler = null;
        try {
            handler = getHandler(id);
        } catch (Exception e) {
            logger.info("error getting handler", e);
            throw e;
        }
        if (handler == null) {
            logger.debug("fail to look up {} because of empty handler", DefaultDataInterpreter.getDataInterpreter().interpretId(id));
            throw new Exception("error getting handler");
        }

        Request.Builder builder = Request.newBuilder();
        Response res = handler.sendRequest(builder.setId(ByteString.copyFrom(id)).setType(RequestType.LOOKUP).setPartition(partition).setCallback(callback));
        if (!res.getSucceeded())
            throw new Exception();
        return res.getData().toByteArray();
    }

    public boolean store(byte[] id, byte[] value, boolean callback) throws  Exception {
        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
        if (isLocalCall(partition)) {
            return PistachiosServer.handler.store(id, partition, value, callback);
        }
        NettyPistachioClientHandler handler = null;
        try {
            handler = getHandler(id);
        } catch (Exception e) {
            logger.info("error getting handler", e);
            return false;
        }
        if (handler == null) {
            logger.debug("fail to look up {} because of empty handler", DefaultDataInterpreter.getDataInterpreter().interpretId(id));
            return false;
        }
        
        Request.Builder builder = Request.newBuilder();
        Response res = handler.sendRequest(builder.setId(ByteString.copyFrom(id)).setType(RequestType.STORE).setPartition(partition).setData(ByteString.copyFrom(value)).setCallback(callback));

        if (res == null) {
            logger.debug("fail");
            return false;
        }
        return res.getSucceeded();
    }

    public boolean processBatch(byte[] id, List<byte[]> events)  throws  Exception {
        NettyPistachioClientHandler handler = null;
        try {
            handler = getHandler(id);
        } catch (Exception e) {
            logger.info("error getting handler", e);
            return false;
        }
        if (handler == null) {
            logger.debug("fail to look up {} because of empty handler", DefaultDataInterpreter.getDataInterpreter().interpretId(id));
            return false;
        }
        Request.Builder builder = Request.newBuilder();
        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
        builder.setId(ByteString.copyFrom(id)).setType(RequestType.PROCESS_EVENT).setPartition(partition);
        for (byte[] event : events) {
            builder.addEvents(ByteString.copyFrom(event));
        }

        if ((conf.getString("pistachio.process.jar.server.url") != null) &&
                (conf.getString("pistachio.process.class") != null)) {
            builder.setJarServerUrl(conf.getString("pistachio.process.jar.server.url"));
            builder.setProcessClass(conf.getString("pistachio.process.class"));
        }

        Response res = handler.sendRequest(builder);
        return res.getSucceeded();
    }

    @Override
    public Map<byte[], byte[]> multiLookup(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        Future<Map<byte[], byte[]>> res = multiLookupAsync(ids, callback);
        long timeoutMs = System.currentTimeMillis() + 100 * 1000L;
        try {
            return res.get(timeoutMs - System.currentTimeMillis(), TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            logger.warn("lookup {} failure ", DefaultDataInterpreter.getDataInterpreter().interpretIds(ids), e);
            throw new RequestTimeoutException("lookup failure: " + e);
        }
    }

    @Override
    public Future<Map<byte[], byte[]>> multiLookupAsync(List<byte[]> ids, boolean callback) throws MasterNotFoundException, ConnectionBrokenException, Exception {
        Map<Long, List<byte[]>> partition2ids = new HashMap<>();
        for (byte[] id : ids) {
            long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition("PistachiosResource"));
            List<byte[]> partitionIds = partition2ids.get(partition);
            if (partitionIds == null) {
                partitionIds = new ArrayList<>();
                partition2ids.put(partition, partitionIds);
            }
            partitionIds.add(id);
        }
        // TODO: change to request per partition, right now request per handler
        Map<NettyPistachioClientHandler, Request.Builder> handler2reqs = new HashMap<>();
        for (Map.Entry<Long, List<byte[]>> entry: partition2ids.entrySet()) {
            NettyPistachioClientHandler handler = null;
            try {
                handler = getHandlerForPartition(entry.getKey());
            } catch (Exception e) {
                logger.info("error getting handler", e);
            }
            if (handler == null) {
                logger.debug("fail to look up {} because of empty handler", entry.getKey());
                continue;
            }
            Request.Builder builder = handler2reqs.get(handler);
            if (builder == null) {
                builder = Request.newBuilder().setType(RequestType.MULTI_LOOKUP).setCallback(callback);
                handler2reqs.put(handler, builder);
            }
            Request.Builder innerRequest = Request.newBuilder().setPartition(entry.getKey());
            for (byte[] value: entry.getValue()) {
                builder.addIds(ByteString.copyFrom(value));
            }
            builder.addRequests(innerRequest);
        }
        final SettableFuture<Map<byte[], byte[]>> ret = SettableFuture.create();
        final Map<byte[], byte[]> respMap = new ConcurrentHashMap<>(handler2reqs.entrySet().size());
        final AtomicInteger latch = new AtomicInteger(handler2reqs.size());

        for (final Map.Entry<NettyPistachioClientHandler, Request.Builder> entry: handler2reqs.entrySet()) {
            NettyPistachioClientHandler handler = entry.getKey();
            ListenableFuture<Response> future = handler.sendRequestAsync(entry.getValue());
            Futures.addCallback(future, new FutureCallback<Response>() {
                @Override
                public void onFailure(Throwable e) {
                    logger.error("caught exception when multi lookup {}", entry.getValue().getPartition(), e);
                    if (latch.decrementAndGet() <= 0) {
                        logger.info("about to set map, on failure");
                        ret.set(respMap);
                    };
                }

                @Override
                public void onSuccess(Response res) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("got response {} for multi-lookup, on success", res);
                    }
                    for (Response re: res.getResponsesList()) {
                        if (re.getSucceeded()) {
                            respMap.put(re.getId().toByteArray(), re.getData().toByteArray());
                        } else {
                            logger.warn("failed to get data for id {}", re.getId());
                        }
                    }
                    if (latch.decrementAndGet() <= 0) {
                        if (logger.isDebugEnabled()) {
                            logger.debug("about to set map for multi-lookup, on success");
                        }
                        ret.set(respMap);
                    }
                }
            });
        }

        return ret;
    }

    private ListenableFuture<Boolean> createBooleanValueFuture(boolean value) {
        SettableFuture<Boolean> future = SettableFuture.create();
        future.set(value);
        return future;
    }

    @Override
    public Future<Boolean> storeAsync(byte[] id, byte[] value)  throws Exception{
        long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
        if (isLocalCall(partition)) {
            return createBooleanValueFuture(PistachiosServer.handler.store(id, partition, value, false));
        }
        NettyPistachioClientHandler handler = null;
        try {
            handler = getHandler(id);
        } catch (Exception e) {
            logger.info("error getting handler", e);
            return createBooleanValueFuture(false);
        }
        if (handler == null) {
            logger.debug("fail to look up {} because of empty handler", id);
            return createBooleanValueFuture(false);
        }
        Request.Builder builder = Request.newBuilder();
        ListenableFuture<Response> res = handler.sendRequestAsync(
                builder.setId(ByteString.copyFrom(id))
                        .setType(RequestType.STORE)
                        .setPartition(partition)
                        .setData(ByteString.copyFrom(value)
                        )
        );

        return Futures.transform(res, BooleanResponseTransformFunction);
    }

    Function<Response, Boolean> BooleanResponseTransformFunction =
            new Function<Response, Boolean>() {
                @Override
                public Boolean apply(Response response) {
                    return response.getSucceeded();
                }
            };

    @Override
    public Future<Map<byte[], Boolean>> multiProcessAsync(Map<byte[], byte[]> events) throws  Exception{
        Map<Long, Request.Builder> partition2reqs = new HashMap<>();
        for (Map.Entry<byte[], byte[]> entry: events.entrySet()) {
            byte[] id = entry.getKey();
            long partition = partitioner.getPartition(id, helixPartitionSpectator.getTotalPartition(RESOURCE));
            Request.Builder pBuilder = partition2reqs.get(partition);
            if (pBuilder == null) {
                pBuilder = Request.newBuilder().setPartition(partition);
                partition2reqs.put(partition, pBuilder);
            }
            pBuilder.addIds(ByteString.copyFrom(id)).addEvents(ByteString.copyFrom(entry.getValue()));
        }

        Map<NettyPistachioClientHandler, Request.Builder> handler2reqs = new HashMap<>();
        for (Map.Entry<Long, Request.Builder> entry: partition2reqs.entrySet()) {
            NettyPistachioClientHandler handler = null;
            try {
                handler = getHandlerForPartition(entry.getValue().getPartition());
            } catch (Exception e) {
                logger.info("error getting handler", e);
            }
            if (handler == null) {
                logger.debug("fail to look up {} because of empty handler", entry.getKey());
                continue;
            }
            Request.Builder builder = handler2reqs.get(handler);
            if (builder == null) {
                builder = Request.newBuilder().setType(RequestType.MULTI_PROCESS_EVENT);
                handler2reqs.put(handler, builder);
            }
            builder.addRequests(entry.getValue());
        }

        final SettableFuture<Map<byte[], Boolean>> ret = SettableFuture.create();
        final Map<byte[], Boolean> respMap = new ConcurrentHashMap<>(handler2reqs.entrySet().size());
        final AtomicInteger latch = new AtomicInteger(handler2reqs.size());

        for (final Map.Entry<NettyPistachioClientHandler, Request.Builder> entry: handler2reqs.entrySet()) {
            NettyPistachioClientHandler handler = entry.getKey();
            ListenableFuture<Response> future = handler.sendRequestAsync(entry.getValue());
            Futures.addCallback(future, new FutureCallback<Response>() {
                @Override
                public void onFailure(Throwable e) {
                    logger.error("caught exception when multi lookup {}", entry.getValue().getPartition(), e);
                    if (latch.decrementAndGet() <= 0) {
                        logger.info("about to set map, on failure");
                        ret.set(respMap);
                    };
                }

                @Override
                public void onSuccess(Response res) {
                    if (logger.isDebugEnabled()) {
                        logger.debug("got response {} for multi-process, on success", res);
                    }
                    for (Response re: res.getResponsesList()) {
                        respMap.put(re.getId().toByteArray(), re.getSucceeded());
                    }
                    if (latch.decrementAndGet() <= 0) {
                        if (logger.isDebugEnabled()) {
                            logger.debug("about to set map for multi-process, on success");
                        }
                        ret.set(respMap);
                    }
                }
            });
        }
        if (logger.isDebugEnabled()) {
            logger.debug("result of multi-process {}", ret);
        }
        return ret;
    }

    static private class TestRunnable extends Thread {
        private java.util.Random rand = new java.util.Random();
        private NettyPistachioClientHandler handler;
        public TestRunnable(NettyPistachioClientHandler handler0) {
            handler = handler0;
        }
        public void run() {
            try {
                // Request and get the response.
                Long l = rand.nextLong();
                System.out.println("sending : " + l);
                Request.Builder builder = Request.newBuilder();
                Response res = handler.sendRequest(builder.setId(ByteString.copyFrom(com.google.common.primitives.Longs.toByteArray(l))));

                if (l != com.google.common.primitives.Longs.fromByteArray(res.getId().toByteArray())) {
                    System.out.println("Thread #" + Thread.currentThread().getId() + ", incorrect result: " + l + ", " + res.getId());
                } else  {
                    System.out.println("Thread #" + Thread.currentThread().getId() + ", correct result: " + l );
                }
            } catch (Exception e) {
                System.out.println("error: " + e);
            }
        }
    }
    public class NettyPistachioIterator implements PistachioIterator {
        protected long id;
        protected int partitionId;
        private Kryo kryo = new Kryo();
        private String lastKey = null;
        private String lastIp = null;
        public String getLastKey() {
            return lastKey;
        }
        public NettyPistachioIterator(int partitionId){
            Random rand = new Random();
            id = rand.nextLong();
            this.partitionId = partitionId;
        }
        @Override
        public KeyValue getNext()  throws MasterNotFoundException, Exception{
            long partition = partitionId;
            if (isLocalCall(partition)) {
                return kryo.readObject(new Input(PistachiosServer.handler.getNext(partition, id)), KeyValue.class);
            }
            NettyPistachioClientHandler handler = null;
            try {
                handler = getHandlerForPartition(partition);
            } catch (Exception e) {
                logger.info("error getting handler", e);
                throw e;
            }
            if (handler == null) {
                logger.debug("fail to get next on verison {}, partition {} because of empty handler",id, partition);
                throw new Exception("error getting handler");
            }
            if(lastIp != null && !handler.getIp().equals(lastIp) && lastKey != null){
                jump(lastKey.getBytes());
                lastIp = handler.getIp();
            }
            Request.Builder builder = Request.newBuilder();
            Response res = handler.sendRequest(builder.setVersionid(id).setType(RequestType.GETNEXT).setPartition(partition));
            logger.info("get response success" +  res.getSucceeded());
            if (!res.getSucceeded())
                throw new Exception();
            Input input = new Input(res.getData().toByteArray());

            KeyValue keyValue = kryo.readObject(input, KeyValue.class);
            lastKey = new String(keyValue.key);
            logger.info("last key "+lastKey);
            return keyValue;
        }

        @Override
        public void jump(byte[] key)  throws MasterNotFoundException, Exception{
            long partition = partitionId;
            if (isLocalCall(partition)) {
                PistachiosServer.handler.jump(key, partition, id);
            }
            NettyPistachioClientHandler handler = null;
            try {
                handler = getHandlerForPartition(partition);
            } catch (Exception e) {
                logger.info("error getting handler", e);
                throw e;
            }
            if (handler == null) {
                logger.debug("fail to jump to {} on verison {}, partition {} because of empty handler",DefaultDataInterpreter.getDataInterpreter().interpretId(key), id, partition);
                throw new Exception("error getting handler");
            }
            Request.Builder builder = Request.newBuilder();
            Response res = handler.sendRequest(builder.setVersionid(id).setId(ByteString.copyFrom(key)).setType(RequestType.JUMP).setPartition(partition));
            if (!res.getSucceeded())
                throw new Exception();
            lastKey = new String(key);
        }
    }
    public static void main(String[] args) throws Exception {
        // Configure SSL.
        final SslContext sslCtx;
        if (SSL) {
            sslCtx = SslContext.newClientContext(InsecureTrustManagerFactory.INSTANCE);
        } else {
            sslCtx = null;
        }
        System.out.println("start test");

        EventLoopGroup group = new NioEventLoopGroup();
        try {
            Bootstrap b = new Bootstrap();
            b.group(group)
                    .channel(NioSocketChannel.class)
                    .handler(new NettyPistachioClientInitializer(sslCtx));

            // Make a new connection.
            Channel ch = b.connect(HOST, PORT).sync().channel();

            // Get the handler instance to initiate the request.
            NettyPistachioClientHandler handler = ch.pipeline().get(NettyPistachioClientHandler.class);

            // Request and get the response.
            //Response res = handler.lookup(12345L);
            Thread [] t = new Thread[1000];
            for (int i = 0; i< 1000; i++) {
                t[i] = new TestRunnable(handler);
            }
            for (int i = 0; i< 1000; i++) {
                t[i].start();
            }
            for (int i = 0; i< 1000; i++) {
                t[i].join();
            }

            // Close the connection.
            ch.close();

            // Print the response at last but not least.
            //System.out.println("response: "+ res.getData());
        } finally {
            group.shutdownGracefully();
        }
    }

    @Override
    public PistachioIterator iterator(long partitionId) {
        return new NettyPistachioIterator((int)partitionId);
    }
}
