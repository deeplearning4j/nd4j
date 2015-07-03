package org.nd4j.paramserver.pistachio.pb.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.OutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.net.InetAddress;

/**
 * Simple library class for working with JNI (Java Native Interface)
 *
 * @see http://frommyplayground.com/how-to-load-native-jni-library-from-jar
 *
 * @author Adam Heirnich <adam@adamh.cz>, http://www.adamh.cz
 */
public class NativeUtils {
    private static Logger logger = LoggerFactory.getLogger(NativeUtils.class);
    private static String hostnameCache = null;

    /**
     * Private constructor - this class will never be instanced
     */
    private NativeUtils() {
    }

    public static String getHostname() {
        if (hostnameCache != null)
            return hostnameCache;

        String ret = null;
        try {
            // Use a ProcessBuilder
            ProcessBuilder pb = new ProcessBuilder("hostname");

            Process p = pb.start();
            InputStream is = p.getInputStream();
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String line = null;
            while ((line = br.readLine()) != null) {
                if (ret == null)
                    ret = line;
            }
            int r = p.waitFor(); // Let the process finish.
            if (r == 0) { // No error
                // run cmd2.
            }
        } catch (Exception e) {
            try {
            ret = InetAddress.getLocalHost().getHostName(); //conf.getString(PROFILE_HELIX_INSTANCE_ID) // instanceName
            } catch (Exception e1) {
                ret = "ERROR";
            }
        }

        if(ret.equals("localhost.localdomain") || hostnameCache != null && hostnameCache.equals("localhost.localdomain"))
            hostnameCache = "127.0.0.1";

        ret = hostnameCache;

        return ret;
    }

    /**
     * Loads library from current JAR archive
     *
     * The file from JAR is copied into system temporary directory and then loaded. The temporary file is deleted after exiting.
     * Method uses String as filename because the pathname is "abstract", not system-dependent.
     *
     * @param filename The filename inside JAR as absolute path (beginning with '/'), e.g. /package/File.ext
     * @throws IOException If temporary file creation or read/write operation fails
     * @throws IllegalArgumentException If source file (param path) does not exist
     * @throws IllegalArgumentException If the path is not absolute or if the filename is shorter than three characters (restriction of {@see File#createTempFile(java.lang.String, java.lang.String)}).
     */
    public static void loadLibraryFromJar(String path) throws IOException {

        if (!path.startsWith("/")) {
            throw new IllegalArgumentException("The path has to be absolute (start with '/').");
        }

        // Obtain filename from path
        String[] parts = path.split("/");
        String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

        // Split filename to prexif and suffix (extension)
        String prefix = "";
        String suffix = null;
        if (filename != null) {
            parts = filename.split("\\.", 2);
            prefix = parts[0];
            suffix = (parts.length > 1) ? "."+parts[parts.length - 1] : null; // Thanks, davs! :-)
        }

        logger.debug("prefix {}, suffix {}", prefix, suffix);

        // Check if the filename is okay
        if (filename == null || prefix.length() < 3) {
            throw new IllegalArgumentException("The filename has to be at least 3 characters long." + filename);
        }

        // Prepare temporary file
        String tempPath = System.getProperty("user.home") + File.separator + filename;
        logger.debug("create lib file {}", tempPath);
        File temp = new File(tempPath);//File.createTempFile(prefix, suffix);
        temp.createNewFile();
        temp.deleteOnExit();

        if (!temp.exists()) {
            throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
        }

        // Prepare buffer for data copying
        byte[] buffer = new byte[1024];
        int readBytes;

        // Open and check input stream
        InputStream is = NativeUtils.class.getResourceAsStream(path);
        if (is == null) {
            throw new FileNotFoundException("File " + path + " was not found inside JAR.");
        }

        // Open output stream and copy data between source file in JAR and the temporary file
        OutputStream os = new FileOutputStream(temp);
        try {
            while ((readBytes = is.read(buffer)) != -1) {
                os.write(buffer, 0, readBytes);
            }
        } finally {
            os.flush();
            // If read/write fails, close streams safely before throwing an exception
            os.close();
            is.close();
        }

        // Finally, load the library
        System.load(temp.getAbsolutePath());
    }
}
