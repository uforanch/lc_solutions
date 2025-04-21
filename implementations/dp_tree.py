class MaxEdgeSet:
    # Function to perform dynamic programming on the tree.
    def dp_on_tree(self, adj, curr, prev, dp, ans):
        for x in adj[curr]:
            if x != prev:
                # Recursively calculate the dp values for each node.
                self.dp_on_tree(adj, x, curr, dp, ans)

                # Update dp[curr][0] by taking the maximum of not selecting and selecting edges.
                dp[curr][0] += max(dp[x][0], dp[x][1])

        for x in adj[curr]:
            if x != prev:
                # Calculate dp[curr][1] using dp values of children.
                dp[curr][1] = max(dp[curr][1], (1 + dp[x][0]) +
                                  (dp[curr][0] - dp[x][0]))#max(dp[x][0], dp[x][1])))

        # Update the global maximum answer with the maximum of dp values for the current node.
        ans[0] = max(ans[0], max(dp[curr][0], dp[curr][1]))

    # Function to solve the problem and find the maximum set of edges.

    def solve(self, n, edges):
        adj = [[] for _ in range(n + 1)]

        # Create an adjacency list to represent the tree.
        for edge in edges:
            x, y = edge
            adj[x].append(y)
            adj[y].append(x)

        # Initialize the answer.
        ans = [0]

        # Initialize the dp array.
        dp = [[0, 0] for _ in range(n + 1)]

        # Start dynamic programming on the tree with the root node as 1.
        self.dp_on_tree(adj, 1, -1, dp, ans)

        # Output the maximum set of edges.
        print(max(dp[1][1], dp[1][0]))
        print(ans[0])
print("max edge set example")
MaxEdgeSet().solve(5, [[1, 2], [1, 3], [3, 4], [3, 5]])


# this was a leetcode...
# https://leetcode.com/problems/sum-of-distances-in-tree/description/
class DistanceSum:
    # Function to perform the first depth-first search to
    # calculate dp1 and sub values.

    def dfs1(self, adj, dp1, dp2, sub, curr, p, n):
        for x in adj[curr]:
            if x != p:
                # Recursively traverse the tree to calculate dp1 and sub values.
                self.dfs1(adj, dp1, dp2, sub, x, curr, n)

                # Update dp1[curr] by adding dp1[x] and sub[x].
                dp1[curr] += dp1[x] + sub[x]

                # Update sub[curr] by adding sub[x].
                sub[curr] += sub[x]

        # Increment sub[curr] to account for the current node itself.
        sub[curr] += 1

    # Function to perform the second depth-first search to
    # calculate dp2 values.

    def dfs2(self, adj, dp1, dp2, sub, curr, p, n):
        if p != -1:
            # Calculate dp2[curr] using dp2 from the parent and sub values.
            dp2[curr] = (dp2[p] - sub[curr]) + (n - sub[curr])
        else:
            # For the root node, dp2 is equal to dp1.
            dp2[curr] = dp1[curr]

        for x in adj[curr]:
            if x != p:
                # Recursively traverse the tree to calculate dp2 values.
                self.dfs2(adj, dp1, dp2, sub, x, curr, n)

    # Function to solve the problem and calculate the sum of
    # distances for each node.

    def solve(self, n, edges):
        adj = [[] for _ in range(n + 1)]

        # Create an adjacency list to represent the tree.
        for edge in edges:
            x, y = edge
            adj[x].append(y)
            adj[y].append(x)

        dp1 = [0] * (n + 1)
        dp2 = [0] * (n + 1)
        sub = [0] * (n + 1)

        # Perform the first depth-first search to calculate dp1 and sub values.
        self.dfs1(adj, dp1, dp2, sub, 1, -1, n)

        # Perform the second depth-first search to calculate dp2 values.
        self.dfs2(adj, dp1, dp2, sub, 1, -1, n)

        # Output the results for each node.
        for i in range(1, n + 1):
            print(dp2[i], end=" ")

print("Distance Sum")
DistanceSum().solve(5, [[1, 2], [1, 3], [3, 4], [3, 5]])