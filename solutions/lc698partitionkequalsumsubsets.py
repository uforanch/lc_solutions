from queue import Queue
from typing import List
import math
from queue import Queue


class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        s = sum(nums)
        n = len(nums)
        if s % k != 0:
            return False
        sp = s // k
        full_clique = (1 << n) - 1
        """
        basic idea is to find all subset sums that are sp.

        doing this MY way: 
            go by SIZE of subset
            travese the power set as a tree
            add children that have subset sum less than target
            add exact sums subsets to a list

            that finds target subsets but what about partitions?
            we're going to make a hashmap of bitmasks to cliques that sum to them
            each time new subset is found add it to clique

            whew let's do it.
        """
        ss_sum = [math.inf] * (1 << n)
        ss_sum[0]=0
        q = Queue()
        q.put(0)
        clique_track = {0}
        while not q.empty():
            v = q.get()
            for i in range(n):
                if (v>>i) % 2 == 0 and ss_sum[v | (1 << i)]==math.inf:
                    v_n = v | (1 << i)
                    ss_sum[v_n] = ss_sum[v] + nums[i]
                    if ss_sum[v_n] < sp:
                        q.put(v_n)

                    if ss_sum[v_n] == sp:
                        track_update = set()
                        for c in clique_track:
                            if not c & v_n:
                                track_update.add(c | v_n)
                        clique_track.update(track_update)
                        if full_clique in clique_track:
                            return True
        return False
def canPartitionKSubsets_dl(nums: List[int], k: int) -> bool:
	total = sum(nums)

	if total % k:
		return False

	reqSum = total // k
	subSets = [0] * k
	nums.sort(reverse = True)

	def recurse(i):
		if i == len(nums):
			return True

		for j in range(k):
			if subSets[j] + nums[i] <= reqSum:
				subSets[j] += nums[i]

				if recurse(i + 1):
					return True

				subSets[j] -= nums[i]

				# Important line, otherwise function will give TLE
				if subSets[j] == 0:
					break

				"""
				Explanation:
				If subSets[j] = 0, it means this is the first time adding values to that subset.
				If the backtrack search fails when adding the values to subSets[j] and subSets[j] remains 0, it will also fail for all subSets from subSets[j+1:].
				Because we are simply going through the previous recursive tree again for a different j+1 position.
				So we can effectively break from the for loop or directly return False.
				"""

		return False

	return recurse(0)

s = Solution()
#s.canPartitionKSubsets([4,3,2,3,5,2,1], 4) # true
#s.canPartitionKSubsets([1,2,3,4], 3) # false
print(s.canPartitionKSubsets([4,4,6,2,3,8,10,2,10,7], 4)) # true
#print(s.canPartitionKSubsets([1,2,1,1,1],3))

from itertools import chain, combinations
print("-----")
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
for s in powerset([4,4,6,2,3,8,10,2,10,7]):
    if sum(s)==14:
        print(s)


