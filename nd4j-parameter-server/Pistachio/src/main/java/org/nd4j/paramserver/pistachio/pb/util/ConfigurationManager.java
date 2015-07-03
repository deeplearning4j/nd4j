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

package org.nd4j.paramserver.pistachio.pb.util;

import org.apache.commons.configuration.CompositeConfiguration;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.configuration.SystemConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Handle all configuration values.
 *
 */
public class ConfigurationManager {
    private static Logger logger = LoggerFactory.getLogger(ConfigurationManager.class);

    public static final String CONFIG_DELIM_REG = "\\.";
    public static final String CONFIG_DELIM = ".";

    private static final CompositeConfiguration cc = new CompositeConfiguration();

    static {
        try {
            // system configuration will be the first in line
            cc.addConfiguration(new SystemConfiguration());

            PropertiesConfiguration pc = new PropertiesConfiguration(new java.io.File(System.getProperty("configPath") + "/pistachios.properties"));
            logger.info("setting base path to {}", System.getProperty("configPath") + "/pistachios.properties");
            cc.addConfiguration(pc);
        }
        catch (ConfigurationException e) {
            logger.error("Failed to load configuration files", e);
        }
    }

    private ConfigurationManager() { }

    public static Configuration getConfiguration() {
        return cc;
    }

}
