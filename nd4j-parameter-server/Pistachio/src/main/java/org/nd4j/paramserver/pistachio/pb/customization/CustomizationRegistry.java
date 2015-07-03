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

package org.nd4j.paramserver.pistachio.pb.customization;
import org.nd4j.paramserver.pistachio.pb.util.ConfigurationManager;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.cache.NodeCache;
import org.apache.curator.framework.recipes.cache.NodeCacheListener;
import org.apache.curator.framework.recipes.cache.ChildData;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.net.URLClassLoader;
import java.net.URL;

public class CustomizationRegistry<T> implements NodeCacheListener{
    private static Logger logger = LoggerFactory.getLogger(CustomizationRegistry.class);

    public static final String PATH = "/pistachio_zk/processor_registry/info";
    private static final String ZOOKEEPER_SERVER = "Pistachio.ZooKeeper.Server";
    protected T processor = null;
    private CuratorFramework    client = null;
    private NodeCache   cache = null;
    private String lastData = null;

    public CustomizationRegistry() {
    }

    protected String getZKPath() {
        return PATH;
    }

    public void init() {
        try
        {
            logger.info("init...");
            client = CuratorFrameworkFactory.newClient(
                ConfigurationManager.getConfiguration().getString(ZOOKEEPER_SERVER),
                new ExponentialBackoffRetry(1000, 3));
            client.start();

            // in this example we will cache data. Notice that this is optional.
            cache = new NodeCache(client, getZKPath());
            cache.start();

            cache.getListenable().addListener(this);

            nodeChanged();

        } catch (Exception e) {
            logger.info("error ", e);
        }
        /*
        finally
        {
            CloseableUtils.closeQuietly(cache);
            CloseableUtils.closeQuietly(client);
        }
        */
    }

    /*
    public static synchronized CustomizationRegistry getInstance() {
        if (instance == null) {
            instance = new CustomizationRegistry();
        }
        return instance;
    }
    */

    public T getCustomization() {
        return processor;
    }

    private void loadFromRegistry(String classURLAndClassName) {
        try {
            String classURL;
            String className;
            classURL = classURLAndClassName.split(";")[0];
            className = classURLAndClassName.split(";")[1];

            URL[] uRLArray = new URL[1];
            uRLArray[0] = new URL(classURL);
            URLClassLoader child = new URLClassLoader(uRLArray, this.getClass().getClassLoader());
            Class classToLoad = Class.forName (className, true, child);
            Object instance = classToLoad.newInstance ();
            processor = (T) instance;
            logger.info("created processor instance based on jar: {} and class: {}", classURL, className);
        } catch (Exception e) {
            logger.info("error loadFromRegistry ", e);
        }
    }

    public synchronized void nodeChanged() throws Exception {
        ChildData data = cache.getCurrentData();

        if (data != null && data.getData() != null) {
            String strData = new String(data.getData());
            logger.info("got data {} from path {}, lastData {}", strData, PATH, lastData);
            if (lastData == null || !lastData.equals(strData)) {
                lastData = strData;
                loadFromRegistry(strData);
            }
        } else {
        logger.info("empty data in path {}", PATH);
        }
    }
}
