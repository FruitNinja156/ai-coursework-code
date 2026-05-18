node_neighbors = {}
def add_nodes(nodelist):
    for node in nodelist:
        if node not in node_neighbors:
            node_neighbors[node] = []

def add_edge(edge):
    u, v = edge
    if v not in node_neighbors[u]:
        node_neighbors[u].append(v)

def depth_first_search(root=None):
    visited = {} 
    order = []
    def dfs(node):
        visited[node] = True
        order.append(node)
        for n in node_neighbors[node]:
            if n not in  visited:
                dfs(n)
    if root:
        dfs(root)
    for node in node_neighbors.keys():
        if  node not in visited:
            dfs(node)
    print ("DFS: ", order)
    return order

def breadth_first_search(root=None):
    
    visited = {} 
    queue = []
    order = []
    def bfs():
        while len(queue) > 0:
            node = queue.pop(0)
            visited[node] = True
            for n in node_neighbors[node]:
                if ( n not in visited) and ( n not in queue):
                    queue.append(n)
                    order.append(n)
    if root:
        queue.append(root)
        order.append(root)
        bfs()
    for node in node_neighbors.keys():
        if  node not in visited:
            queue.append(node)
            order.append(node)
            bfs()
    print ("BFS: ", order)
    return order


if __name__ == '__main__':
    add_nodes([i+1 for i in range(9)])
    add_edge((1,2))
    add_edge((1,4))
    add_edge((2, 3))
    add_edge((2, 5))
    add_edge((2,7))
    add_edge((3, 1))
    add_edge((3,6))
    add_edge((4,6))
    add_edge((5,7))
    add_edge((5,8))
    add_edge((6,8))
    add_edge((7,8))
    add_edge((7,9))
    add_edge((8,9))


    print ("Nodes:", node_neighbors.keys())
    order = depth_first_search(1)
    order = breadth_first_search(1) 
    
  