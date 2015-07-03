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

import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.protobuf.ProtobufDecoder;
import io.netty.handler.codec.protobuf.ProtobufEncoder;
import io.netty.handler.codec.protobuf.ProtobufVarint32FrameDecoder;
import io.netty.handler.codec.protobuf.ProtobufVarint32LengthFieldPrepender;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import java.util.concurrent.TimeUnit;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.ssl.SslContext;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NettyPistachioClientInitializer extends ChannelInitializer<SocketChannel> {

    private final SslContext sslCtx;
    private static Logger logger = LoggerFactory.getLogger(NettyPistachioClientInitializer.class);

    public NettyPistachioClientInitializer(SslContext sslCtx) {
        this.sslCtx = sslCtx;
    }

    @Override
    public void initChannel(SocketChannel ch) {
        ChannelPipeline p = ch.pipeline();
        if (sslCtx != null) {
            p.addLast(sslCtx.newHandler(ch.alloc(), NettyPistachioClient.HOST, NettyPistachioClient.PORT));
        }

        LogLevel level = LogLevel.DEBUG;


        p.addLast(new LoggingHandler(level));
        p.addLast(new ReadTimeoutHandler(ConfigurationManager.getConfiguration().getInt("Network.Netty.ClientReadTimeoutMillis",10000), TimeUnit.MILLISECONDS));
        p.addLast(new ProtobufVarint32FrameDecoder());
        p.addLast(new ProtobufDecoder(NettyPistachioProtocol.Response.getDefaultInstance()));

        p.addLast(new ProtobufVarint32LengthFieldPrepender());
        p.addLast(new ProtobufEncoder());

        p.addLast(new NettyPistachioClientHandler());
    }
}
