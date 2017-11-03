package org.nd4j.autodiff.graph.api;

import lombok.Data;
import org.nd4j.linalg.collection.IntArrayKeyMap;

import java.util.Arrays;

/** Edge in a graph. 
 * May be a directed or undirected edge.<br>
 * Parametrized,
 * and may store a 
 * value/object associated with the edge
 */
@Data
public class Edge<T> {

    private  int[] from;
    private  int[] to;
    private  T value;
    private  boolean directed;


    public Edge(int[] from, int[] to, T value, boolean directed) {
        this.from = new IntArrayKeyMap.IntArray(from).getBackingArray();
        this.to = new IntArrayKeyMap.IntArray(to).getBackingArray();
        this.value = value;
        this.directed = directed;
    }


    @Override
    public String toString() {
        return "edge(" + (directed ? "directed" : "undirected") + "," + Arrays.toString(from) + (directed ? "->" : "--") + Arrays.toString(to) + ","
                        + (value != null ? value : "") + ")";
    }

}
