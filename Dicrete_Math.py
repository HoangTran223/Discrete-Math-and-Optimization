""" Breadth First Search (BFS) for a Graph:"""
from collections import defaultdict     # Là 1 từ điển cung cấp các giá trị mặc định cho các khóa không tồn tại
                                        # Tức 1 khóa ko tồn tại, nhưng từ điển vẫn trả về 1 gtri mặc định mình đặt, ko ra error như từ điển thường

class Graph():
    def __init__(self):
        self.graph = defaultdict(list)
    # Function to add edge to graph:
    def add_edge(self, u, v):
        self.graph[u].append(v)
    

    # Function to print BFS:
    def BFS(self, s):

        # Ví dụ: ta có (5, 4), (5, 7), (5, 8), (5, 9), (5, 11), (5, 11) với 5 là max của keys
        # Thì max(neighbors) sẽ trả về 2 giá trị 11, và ta sẽ cần max(max(neighbors)) để cho nó trả về đúng 1 giá trị
        max_vertex = max(max(self.graph.keys()), max((max(neighbors) for neighbors in self.graph.values())) + 1)
        visited = [False] * (max_vertex + 1)
        queue = []
        queue.append(s)
        visited[s] = True

        while queue:
            s = queue.pop(0)
            print(s, end = ' ')
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
g = Graph()
g.add_edge(0,1)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(1,4)
g.add_edge(2,5)
g.add_edge(2,6)
g.add_edge(5,7)
g.add_edge(3,8)
g.add_edge(3,9)
g.BFS(1)


"""Depth First Search (DFS) for a graph:"""
class Graph():
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
    
    def DFS(self, s):
        visited = set()
        visited.add(s)
        print(s, end = ' ')
        for neighbors in self.graph[s]:
            if neighbors not in visited:
                self.DFS(neighbors)

g = Graph()
g.add_edge(0,1)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(1,4)
g.add_edge(2,5)
g.add_edge(2,6)
g.add_edge(5,7)
g.add_edge(3,8)
g.add_edge(3,9)
g.add_edge(6,11)
g.add_edge(6,13)
g.add_edge(7,10)
g.DFS(1)



""" Bài toán cây khung nhỏ nhất:"""
# THUẬT TOÁN KRUSKAL: ở đây ta sử dụng cấu trúc dữ liệu disjoint sets

class Graph():
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = []

    def add_edge(self, u, v, weight):
        self.graph.append([u, v, weight])

    # Hàm này tìm gốc của tập hợp chứa đỉnh i 
    def find(self, parent, i):
        if parent[i] != i:                              # Nếu i không là gốc của tập hợp chứa nó
            parent[i] = self.find(parent, parent[i])
        return parent[i]
    
    # Hàm hợp nhất 2 tập có gốc là x và y:
    def union(self, parent, rank, x, y):
        # Cây có độ cao (rank) nhỏ trở thành con cây có rank cao hơn, để cây không tăng độ cao
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1


    def kruskal(self):
        result = []
        i = 0           # Used for sorted edges
        e = 0           # Used for result[]
        self.graph = sorted(self.graph, key = lambda x: x[2])     # Sort of weight

        parent = []                                               # Tập các node         
        rank = []
        for node in range(self.vertices + 1):                     # Bởi vì parent phải chứa cả đỉnh cuối là self.vertices do hàm find_parent   
            parent.append(node)
            rank.append(0)
        
        while e < self.vertices - 1:            # When |edges| < |vertices| - 1
            u, v, w = self.graph[i]             # The smallest edge
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:                          # Nếu gốc khác nhau, tức nó thuộc 2 tập hợp khác nahu              
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        
        kruskal_cost = 0
        print("Edges in the MST: ")
        for u, v, weight in result:
            kruskal_cost += weight
            print(f"{u} -- {v} with weight: {weight}")
        print("Cost of MST:", kruskal_cost)

g = Graph(7)
# g.add_edge(0, 1, 2) cũng được do code trên cho cả đỉnh 0
g.add_edge(1, 2, 8)
g.add_edge(1, 3, 5)
g.add_edge(2, 3, 10)
g.add_edge(2, 4, 2)
g.add_edge(2, 5, 18)
g.add_edge(4, 3, 3)
g.add_edge(4, 6, 20)
g.add_edge(4, 7, 14)
g.add_edge(3, 6, 16)
g.add_edge(4, 5, 12)
g.add_edge(7, 5, 4)
g.add_edge(7, 6, 26)
g.kruskal()





# Thuật toán Prim: 
""" Thuật toán:
- Tạo 1 tập mstSet để theo dõi đỉnh của MST
- Khởi tạo tất cả khóa là INFINITE (MAXSIZE)
- Gán key value = 0 cho đỉnh đầu tiên
- While mstSet <= n:
+ u ko thuộc mstSet, có giá trị key nhỏ nhất
+ Update key value của tất cả đỉnh kề u: lặp qua tất cả đỉnh v kề với u:
	Nếu w(u,v) nhỏ hơn key value trc đó của v, 
	Update key value đó = w(u,v)
    build_mst[v] = u                """

import sys
class Graph():
    def __init__(self, verticles):
        self.verticles = verticles
        self.graph = [[0 for i in range(verticles)]
                            for j in range(verticles)]
    
    def printMST(self, build_mst):
        print("The Prim Algorithm: ")
        for i in range(1, self.verticles):                           # Để tránh in giá trị đỉnh i -- đỉnh i: w = 0
            print(f"{build_mst[i] + 1} -- {i+1} with the weight: {self.graph[build_mst[i]][i]}")
    
    def min_key_index(self, keys, mst_set):
        min = sys.maxsize                                            # Trả về INT_MAX
        min_index = -1
        for v in range(self.verticles):
            if keys[v] < min and mst_set[v] == False:                    # Lấy đỉnh có trọng số đến các cạnh thấp nhất và chưa được chọn vào MST
                min = keys[v] 
                min_index = v
        return min_index
    
    def Prim(self):
        keys = [sys.maxsize] * self.verticles                        # Tập để lưu trọng số thấp nhất của các đỉnh
        build_mst = [None] * self.verticles                          # Mảng để lưu MST 
        mst_set = [False] * self.verticles                           # Đây về sau là tập các đỉnh chưa được chọn vào MST

        keys[0] = 0
        build_mst[0] = 0
        for i in range(self.verticles):
            u = self.min_key_index(keys, mst_set)                                
            mst_set[u] = True
            for v in range(self.verticles):
                if (self.graph[u][v] > 0 and                     # Tức nó không phải TH (i, i)      
                            mst_set[v] == False and              # Các đỉnh chưa được thêm vào MST
                                keys[v] > self.graph[u][v]):      
                    keys[v] = self.graph[u][v] 
                    build_mst[v] = u
        self.printMST(build_mst)

g = Graph(7)
g.graph = [[0, 8, 5, sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize],
            [8, 0, 10, 2, 18, sys.maxsize, sys.maxsize],
            [5, 10, 0, 3, sys.maxsize, 16, sys.maxsize],
            [sys.maxsize, 2, 3, 0, 12, 30, 14],
            [sys.maxsize, 18, sys.maxsize, 12, 0, sys.maxsize, 4],
            [sys.maxsize, sys.maxsize, 16, 30, sys.maxsize, 0, 26],
            [sys.maxsize, sys.maxsize, sys.maxsize, 14, 4, 26, 0]]
g.Prim()





""" BÀI TOÁN TÌM ĐƯỜNG ĐI NGẮN NHẤT:"""

# Cho trước khoảng cách giữa từ đỉnh s đến đỉnh t. Tìm đường đi ngắn nhất từ s đến t
import sys
weight_matrix = [[0, 6, 22, 8, 40],
                    [6, 0, 15, sys.maxsize, 5],
                    [22, 15, 0, sys.maxsize, sys.maxsize],
                    [8, sys.maxsize, sys.maxsize, 0, 26],
                    [40, 5, sys.maxsize, 26, 0]]
start_verticle = 0
s = int(input())                # Nếu ta nhập 1 sẽ bị lỗi
end_verticle = s - 1
distance_start_verticle = [0, 6, 21, 8, 34]
min_distance = []
i = end_verticle
while i != start_verticle:
    for u in range(len(distance_start_verticle)):
        if distance_start_verticle[i] == distance_start_verticle[u] + weight_matrix[i][u]:
            min_distance.append(i)
            i = u
            break

if min_distance[0] != end_verticle :
    min_distance.append(0)
    min_distance.append(end_verticle)
else:
    min_distance.append(0)

for x in min_distance:
    print(x + 1)





# Thuật toán Dijkstra: với độ phức tạp O(n^2):

import sys
class Graph():
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[0 for i in range(self.vertices)]
                        for j in range(self.vertices)]
    
    def min_index(self, djset, weights):
        min = sys.maxsize
        min_index = -1 
        for v in range(self.vertices):
            if weights[v] < min and djset[v] == False:
                min = weights[v]
                min_index = v 
        return min_index
    
    def print_MST(self, parent, weights):
        print("Vertex \t Distance \tPath")
        for u in range(self.vertices):
            print(u+1, "\t", weights[u], "\t\t", end ="")
            self.print_path(parent, u)
            print()
    

        
    def print_path(self, parent, j):
        if parent[j] == -1:
            print(j + 1, end="")
            return
        self.print_path(parent, parent[j])
        print(" ->", j + 1, end="")
        


    def dijkstra(self, start):
        djset = [False]*self.vertices
        parent = [-1]*self.vertices
        weights = [sys.maxsize]*self.vertices

        weights[start] = 0

        for i in range(self.vertices):
            u = self.min_index(djset, weights)
            djset[u] = True
            for v in range(self.vertices):
                if self.graph[u][v] > 0 and djset[v] == False and weights[v] > weights[u] + self.graph[u][v]:
                    weights[v] = weights[u] + self.graph[u][v]
                    parent[v] = u 
        
        self.print_MST(parent, weights)

g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
               [4, 0, 8, 0, 0, 0, 0, 11, 0],
               [0, 8, 0, 7, 0, 4, 0, 0, 2],
               [0, 0, 7, 0, 9, 14, 0, 0, 0],
               [0, 0, 0, 9, 0, 10, 0, 0, 0],
               [0, 0, 4, 14, 10, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 1, 6],
               [8, 11, 0, 0, 0, 0, 1, 0, 7],
               [0, 0, 2, 0, 0, 0, 6, 7, 0]
               ]
g.dijkstra(0)

# g = Graph(5)
# g.graph = [[0, sys.maxsize, 1, sys.maxsize, sys.maxsize],
#             [3, 0, 5, sys.maxsize, sys.maxsize],
#             [sys.maxsize, sys.maxsize, 0, 2, 4],
#             [5, sys.maxsize, sys.maxsize, 0, sys.maxsize],
#             [sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize, 0]
#             ]
# g.dijkstra(1)






# Thuật toán Dijkstra sử dụng Heapsort với độ phức tạp O(mlogn):
import heapq
import sys

class Graph():
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = []

    def add_edge(self, u, v, weight):
        self.graph.append([u - 1, v - 1, weight])

    def heap_dijkstra(self, start_verticle):
        start = start_verticle - 1
        pq = []
        heapq.heappush(pq, (0, start))  # Khởi tạo đỉnh nguồn
        keys = [sys.maxsize] * self.vertices
        keys[start] = 0
        prev = [None] * self.vertices  # Mảng lưu trữ đỉnh trước đỉnh hiện tại

        while pq:
            distance, u = heapq.heappop(pq)  # Lấy đỉnh u có khoảng cách ngắn nhất
            for edge in self.graph:  # Lặp qua tất cả các đỉnh kề với u, và update distance nếu có đường đi ngắn hơn
                if edge[0] == u:
                    v = edge[1]
                    weight = edge[2]

                    if keys[v] > keys[u] + weight:
                        keys[v] = keys[u] + weight
                        heapq.heappush(pq, (keys[v], v))
                        prev[v] = u

        print("Verticles\t Distances\t Path")
        for i in range(self.vertices):
            print(i + 1, "\t\t", keys[i], "\t\t", end="")
            self.print_path(prev, start, i)
            print()

    def print_path(self, prev, start, end):
        if end == start:
            print(end + 1, end="")
        elif prev[end] is None:
            print("No path from {} to {}".format(start + 1, end + 1))
        else:
            self.print_path(prev, start, prev[end])
            print(" -> {}".format(end + 1), end="")


# g = Graph(5)
# g.add_edge(1, 3, 1)
# g.add_edge(2, 1, 3)
# g.add_edge(2, 3, 5)
# g.add_edge(3, 4, 2)
# g.add_edge(3, 5, 4)
# g.add_edge(4, 1, 5)
# g.heap_dijkstra(2)

g = Graph(9)
g.add_edge(1, 2, 4)
g.add_edge(1, 8, 8)
g.add_edge(2, 3, 8)
g.add_edge(2, 8, 11)
g.add_edge(3, 4, 7)
g.add_edge(3, 9, 2)
g.add_edge(3, 6, 4)
g.add_edge(4, 5, 9)
g.add_edge(4, 6, 14)
g.add_edge(5, 6, 10)
g.add_edge(6, 7, 2)
g.add_edge(7, 8, 1)
g.add_edge(7, 9, 6)
g.add_edge(8, 9, 7)
g.heap_dijkstra(1)


# Tìm đường đi ngắn nhất giữa 2 đỉnh bất kỳ sử dụng thuật toán Floyd Warshall với O(n^3):
# Thuật toán này làm việc được với cả trọng số âm, nhưng không với chu trình âm
""" Thuật toán Floyd:
Ta sẽ cập nhật d[i, j] = min( d[i, j], d[i, k] + d[k, j] ), trong đó
i, j: điểm đầu, cuối; k: đỉnh trung gian 
"""
import sys
class Graph():
    def __init__(self, vertices, dis, check_dis):
        self.vertices = vertices
        self.dis = dis
        self.check_dis = check_dis                              # Mảng theo dõi đỉnh trung gian, a[i][j] là đỉnh trung gian
                                                                # của đường đi ngắn nhất từ i đến j
        self.graph = [[0 for i in range(vertices)]
                        for j in range(vertices)]
    
    def initialise(self):
        for i in range(self.vertices):
            for j in range(self.vertices):
                self.dis[i][j] = self.graph[i][j]               # Copy bảng
            
                if self.graph[i][j] == sys.maxsize:             # Tức ko có đường đi từ i đến j
                    self.check_dis[i][j] = -1
                else:
                    self.check_dis[i][j] = j

    def floyd_warshall(self):
        for k in range(self.vertices):
            for i in range(self.vertices):
                for j in range(self.vertices):
                    if self.dis[i][k] == sys.maxsize or self.dis[k][j] == sys.maxsize:              # If edge doesn't exist
                        continue
                    if self.dis[i][j] > self.dis[i][k] + self.dis[k][j]:                            # Tìm được đường đi ngắn hơn giữa i và j thông qua nút trung gian k   
                        self.dis[i][j] = self.dis[i][k] + self.dis[k][j]                                
                        self.check_dis[i][j] = self.check_dis[i][k]                                 # Đỉnh trung gian giữa i, j thay bằng giữa i, k

    def build_path(self, u, v):
        if self.check_dis[u][v] == -1:
            return []
        else:
            path = [u]
            while u != v:
                u = self.check_dis[u][v]                # Liệt kê tất cả đỉnh trung gian cho đến đỉnh gốc
                path.append(u)
            return path

def print_shortest_path(path):
    if not path:
        print("No path exists")
        return
    for i in range(len(path) - 1):
        print(path[i], end = " -> ")
    print(path[-1])

INF = sys.maxsize
graph = [ [ 0, 3, INF, 7 ],
            [ 8, 0, 2, INF ],
            [ 5, INF, 0, 1 ],
            [ 2, INF, INF, 0 ] ]
dis = [[INF for i in range(4)] 
        for i in range(4)]
check_dis = [[-1 for i in range(4)]
        for i in range(4)]

g = Graph(4, dis, check_dis)
g.graph = graph
g.initialise()
g.floyd_warshall()

path = g.build_path(1, 3)
print_shortest_path(path)
path = g.build_path(3, 2)
print_shortest_path(path)


