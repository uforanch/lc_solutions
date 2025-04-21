import heapq
import math
"""
TODO:
start with bfs, dfs, topological sort

get to bellmand ford and all that

difficulties: making examples
"""
from collections import deque
from typing import List, Optional
class Graph:
    """
    TODO: add underscores
    """
    def __init__(self, nodes: int, edges: List[List[int]], weights: Optional[List[int]]=None, directed=False):
        self.nodes=nodes
        self.adj=[[] for _ in range(nodes)]

        self.weights = dict()
        self.degrees = [[0,0] for _ in range(nodes)] if directed else [0 for _ in range(nodes)]
        self.directed = directed
        if weights is None:
            weights=[1 for _ in edges]
        for i, edge in enumerate(edges):
            self.adj[edge[0]].append(edge[1])
            self.weights[(edge[0], edge[1])] = weights[i]
            if not directed:
                self.adj[edge[1]].append(edge[0])
                self.weights[(edge[1], edge[0])] = weights[i]
                self.degrees[edge[0]]+=1
                self.degrees[edge[1]]+=1
            else:
                self.degrees[edge[0]][1]+=1
                self.degrees[edge[1]][0]+=1
        for node in range(self.nodes):
            self.adj[node].sort(key=lambda x : self.weights[(node, x)])

    def getNodes(self):
        return self.nodes

    def getAdj(self, node: int):
        return tuple(self.adj[node])

    def getWeight(self, node_s: int, node_e: int):
        e = (node_s, node_e)
        return self.weights[e] if e in self.weights.keys() else None

    def getDegree(self, node: int):
        return self.degrees[node] if not self.directed else tuple(self.degrees[node])



#list TODO
# disjoint set TESTS, timing
# various dfs (top sort), bfs, kahn
# dijsktra TESTS with timing
# floyd-warshall, bellman-ford
# kruskall, prim

def BFS(graph: Graph, start: int):
    node_deq = deque([start])
    travelled = []
    travelled_set = set()
    while node_deq:
        node = node_deq.pop()

        for adj_node in graph.getAdj(node):
            if adj_node not in travelled_set:
                node_deq.append(adj_node)
        travelled.append(node)
        travelled_set.add(node)
    return travelled



def DFS(graph: Graph, start: int):
    pre_ord=[]
    post_ord=[]
    discovery_set=set()

    def DFS_recur(node: int, stack: List[int]):
        pre_ord.append(node)
        discovery_set.add(node)
        for adj_node in graph.getAdj(node):
            if adj_node in stack:
                print(f"cycle ending in {adj_node} detected")
            if adj_node not in discovery_set:
                DFS_recur(adj_node, stack+[node])
        post_ord.append(node)
    DFS_recur(start, [])
    print(pre_ord)
    print(post_ord)



def kahn(graph: Graph):
    degrees = [list(graph.getDegree(i)) for i in range(graph.getNodes())]
    node_deq = deque([i for i in range(graph.getNodes()) if degrees[i][0]==0])
    out = []
    while node_deq:
        node = node_deq.popleft()
        out.append(node)
        for adj_node in graph.getAdj(node):
            degrees[adj_node][0]-=1
            if degrees[adj_node][0] == 0:
                node_deq.append(adj_node)
            elif degrees[adj_node][0] == -1:
                print(f"cycle ending in {adj_node} detected")
    return out




def disjikstra(graph: Graph, start: int, end: int, replaceHeap=False):
    unvisited = set(range(graph.getNodes()))
    dist = [math.inf for _ in range(graph.getNodes())]
    dist[start]=0
    prev_node = [-1 for _ in range(graph.getNodes())]
    Q = [(0, start)]
    while unvisited:
        _, node = heapq.heappop(Q)
        if node not in unvisited:
            continue
        for adj_node in graph.getAdj(node):
            if adj_node not in unvisited:
                continue
            w=graph.getWeight(node, adj_node)
            prev = dist[adj_node]
            dist[adj_node] = min(dist[node]+w, dist[adj_node])
            if prev != dist[adj_node]:
                prev_node[adj_node] = node
                if replaceHeap:
                    Q[Q.index((dist, adj_node))] = heapq.heappop(Q)
                    heapq.heapify(Q)

                heapq.heappush((dist[adj_node], adj_node))
        unvisited.remove(node)
    past_dist = 0
    n = end
    path = ""
    while True:
        path = str(n) + " " + path
        past_dist += dist[n]
        if n==start:
            break
        n = prev_node[n]
        if n==-1:
            break
    print(path)
    print(past_dist)

def floydWarshall(graph: Graph):

    dist = [[math.inf for _ in range(graph.getNodes())] for _ in range(graph.getNodes())]
    prev = [[None for _ in range(graph.getNodes())] for _ in range(graph.getNodes())]
    for node in range(graph.getNodes()):
        dist[node][node]=0
        prev[node][node] = node
        for adj_node in graph.getAdj(node):
            dist[node][adj_node] = graph.getWeight(node, adj_node)
            prev[node][adj_node] = node
    for k in range(graph.getNodes()):
        for i in range(graph.getNodes()):
            for j in range(graph.getNodes()):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist, prev

def bellmanFord(graph: Graph, start: int):
    




class tests:
    """
    TODO: add asserts later

    """
    def graphTests(self):
        g1_edgelist = [[0,2],[0,1], [2,1]]
        graph1 = Graph(nodes = 3, edges=g1_edgelist, weights=[3,2,1])
        print("graph1")
        for i in range(3):
            print(i, ":", graph1.getAdj(i), graph1.getDegree(i))
        for e in g1_edgelist:
            print(e, graph1.getWeight(e[0], e[1]))
            print((e[1], e[0]), graph1.getWeight(e[1], e[0]))


        graph2 = Graph(nodes = 3, edges=g1_edgelist, weights=[3,2,1], directed=True)
        print("graph2")
        for i in range(3):
            print(i, ":", graph2.getAdj(i), graph2.getDegree(i))
        for e in g1_edgelist:
            print(e, graph2.getWeight(e[0], e[1]))
            print((e[1], e[0]), graph2.getWeight(e[1], e[0]))
t =tests()
t.graphTests()
