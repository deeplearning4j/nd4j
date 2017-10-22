package org.nd4j.graph.intermediate;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

import java.util.List;

/**
 * This class is basic scope representation: as ordered list of ops
 * @author raver119@gmail.com
 */
@EqualsAndHashCode
public class TScope {
    @Getter private List<TNode> nodes;
    @Getter private int id;
    @Getter private String name;


    public TScope(int id, @NonNull String name) {
        this.id = id;
        this.name = name;
    }

    /**
     * This method adds operation to the end of the scope
     *
     * @param node
     */
    public void addNode(@NonNull TNode node) {
        nodes.add(node);
    }

    /**
     * This method returns last node of this scope
     * @return
     */
    public TNode lastNode() {
        return nodes.get(nodes.size() - 1);
    }

    /**
     * This method returns number of nodes in this scope
     *
     * @return
     */
    public int size() {
        return nodes.size();
    }
}
