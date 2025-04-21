from typing import List
from collections import Counter


class Solution:
    def prime_test(self, prime_list, num):
        if num in prime_list:
            return True
        if num < max(prime_list):
            return False
        test_prime = max(prime_list)
        while test_prime<=num:
            test_prime += 2
            for p in prime_list:
                if p > test_prime ** .5:
                    prime_list.append(test_prime)
                    break
                if test_prime % p == 0:
                    break
        if num in prime_list:
            return True
        return False

    def mostFrequentPrime(self, mat: List[List[int]]) -> int:
        prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        p_count = Counter()
        rows = len(mat)
        cols = len(mat[0])

        max_count = 0
        max_count_prime = -1
        for row in range(len(mat)):
            for col in range(len(mat[0])):
                for dir_r, dir_c in [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1], [1, -1], [-1, 1]]:
                    rrow = row + dir_r
                    ccol = col + dir_c
                    num = mat[row][col]
                    while rrow < rows and ccol < cols and rrow >= 0 and ccol >= 0:
                        num = num * 10 + mat[rrow][ccol]
                        if self.prime_test(prime_list, num):
                            p_count[num] += 1
                            if p_count[num] > max_count:
                                max_count = p_count[num]
                                max_count_prime = num
                        rrow = rrow + dir_r
                        ccol = ccol + dir_c
        print(p_count)
        print(prime_list)
        return max_count_prime

#[[9,3,8],[4,2,5],[3,8,6]]
#ouput 43 expected 823


print(Solution().mostFrequentPrime([[1,1],[9,9],[1,1]]))
#print(Solution().mostFrequentPrime([[9,3,8],[4,2,5],[3,8,6]]))