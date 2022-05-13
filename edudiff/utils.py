from collections import Counter
from typing import Callable, List, Any

__all__ = ["topological_order", "gmap"]

def gmap(root: Any, neighbors: Callable, f: Callable):
    """Apply f to every node in the graph defined by [root, neighbors]."""
    if not root:
        return

    seen = {id(root),}
    frontier = [root]
    while frontier:
        node = frontier.pop()
        f(node)

        for neighbor in neighbors(node):
            if id(neighbor) not in seen:
                seen.add(id(neighbor))
                frontier.append(neighbor)


def topological_order(root: Any, neighbors: Callable) -> List[Any]:
    """Construct a topological order starting from `root`.
    
    The edge relation is given by `neighbors`.

    Following the textbook construction, we walk the graph to count the 
    in-degree of each node via a DFS. Then we consme the graph by taking 
    the frontier consisting of nodes with in-degree zero.
    """
    in_links = Counter()

    frontier = [root]
    while frontier:
        node = frontier.pop()

        for neighbor in neighbors(node):
            if id(neighbor) not in in_links:
                frontier.append(neighbor)

            in_links[id(neighbor)] += 1

    topo_order = []
    frontier = [root]
    while frontier:
        node = frontier.pop()
        topo_order.append(node)

        for neighbor in neighbors(node):
            in_links[id(neighbor)] -= 1
            if in_links[id(neighbor)] == 0:
                frontier.append(neighbor)

    return topo_order

