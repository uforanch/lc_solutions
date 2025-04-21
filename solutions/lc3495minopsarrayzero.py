from collections import Counter
import math
from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        """
        honestly going with gut here

        ith item just needs at least X ops on it
        get that for the array

        note - queries are always going to be seperate,
        and always going to be ranges.
        """
        out = 0
        for q in queries:
            op_count_dict = Counter()
            ops = 0
            min_ops = None
            max_ops = 0

            s = q[0]
            # num of ops = highest base 4 digit +1
            while 4 ** ops <= q[1]:
                if 4 ** ops > s:
                    op_count_dict[ops] += 4 ** ops - s
                    s = 4 ** ops
                ops += 1
            op_count_dict[ops] += q[1] - s + 1
            print(op_count_dict)

            odds = 0
            op_count = 0
            for ops in op_count_dict:
                rect = ops * op_count_dict[ops]
                if rect % 2 == 1:
                    odds += 1
                op_count += rect // 2
            op_count += int(math.ceil(odds / 2))
            out += op_count
        return out

Solution().minOperations([[13,16]])