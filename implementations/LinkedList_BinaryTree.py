from collections import defaultdict, Counter
from itertools import product
from typing import List, Optional
import math

""""



"""
class ListNode:
    def __init__(self, val, next):
        self.val = val
        self._next = next

    def __str__(self):
        return f"_{self.val}_"

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, other):
        print(f"{self} set to {other}")
        self._next = other


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self._left = left
        self._right = right

    def __str__(self):
        return f"__{self.val}__"

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, other):
        print(f"{self} left set to {other}")
        self._left = other

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, other):
        print(f"{self} right set to {other}")
        self._right = other

    """

    formula for index of node:
    depth index starts with 0 
    node address translates to left to right placement: lll is 000 on depth 3, which has 8 nodes
    2^depth-1 + placement

    """

    @staticmethod
    def index_to_depth(i):
        # test first
        if i == 0:
            return 0
        return int(math.log(i + 1, 2))

    @staticmethod
    def index_child(i, side):
        d = TreeNode.index_to_depth(i)
        if d == 0:
            return {"l": 1, "r": 2}[side]
        i_placement = i - (2 ** d - 1)
        return (2 ^ (d + 1) - 1) + (i_placement * 2 + {"l": 0, "r": 1}[side])

    @staticmethod
    def child_index(i):
        if i == 0:
            return None, None
        d = TreeNode.index_to_depth(i)
        i_placement = (i - (2 ** d - 1))
        i_bit = i_placement % 2
        i_side = ["l", "r"][i_bit]
        parent_index = (2 ** (d - 1) - 1) + int(i_placement / 2)
        return parent_index, i_side

    def to_list(self):
        # for the "tests" in leetcode where root is expressed as list

        list_out = []
        node_stack = [(self, 0)]
        while node_stack:
            node_current, node_current_index = node_stack.pop()
            list_out_len = len(list_out)
            if node_current_index >= list_out_len:
                if list_out_len == 0:
                    list_out = [None]
                else:
                    list_out += [None] * list_out_len
            list_out[node_current_index] = node_current.val
            # note to self, make a child iterator maybe
            if node_current.left is not None:
                node_stack.append((node_current.left, TreeNode.index_child(node_current_index, "l")))
            if node_current.right is not None:
                node_stack.append((node_current.right, TreeNode.index_child(node_current_index, "r")))

    @staticmethod
    def from_list(l):
        l += [None] * (2 ^ (TreeNode.index_to_depth(len(l) + 1) - 1) - len(l))
        tree_list = [None] * len(l)
        for i, val in enumerate(l):
            if val is None:
                continue
            tree_list[i] = TreeNode(val)
            if i == 0:
                continue
            parent_i, side = TreeNode.child_index(i)
            parent_node = tree_list[parent_i]
            if side == "l":
                parent_node.left = tree_list[i]
            else:
                parent_node.right = tree_list[i]
        return tree_list[0]


def list_to_linked_list(ll):
    prev = None
    for l in reversed(ll):
        node = ListNode(l, prev)
        prev = node
    return node


def test_treenode():
    assert TreeNode.index_to_depth(0) == 0

    assert TreeNode.index_to_depth(1) == 1
    assert TreeNode.index_to_depth(2) == 1
    assert TreeNode.index_to_depth(5) == 2
    assert TreeNode.index_to_depth(6) == 2

    assert TreeNode.index_to_depth(7) == 3

    assert TreeNode.index_child(0, "l") == 1
    assert TreeNode.index_child(0, "r") == 2
    assert TreeNode.index_child(1, "l") == 3
    assert TreeNode.index_child(1, "r") == 4

    assert TreeNode.index_child(2, "l") == 5
    assert TreeNode.index_child(2, "r") == 6

    assert TreeNode.child_index(1) == (0, "l")
    assert TreeNode.child_index(2) == (0, "r")
    assert TreeNode.child_index(3) == (1, "l")
    assert TreeNode.child_index(4) == (1, "r")
    assert TreeNode.child_index(5) == (2, "l")
    assert TreeNode.child_index(6) == (2, "r")

    return
    t1 = TreeNode.from_list([0, 1, 2, 3, 4, 5, 6, 7])
    assert t1.val == 0
    assert t1.left.val == 1
    assert t1.left.to.list() == [1, 3, 4]


head = list_to_linked_list([1, 2, 3, 4, 5])
example1 = ["ABABA", 2]
example2 = ["AABABBA", 2]
# print(Solution().characterReplacement("AABABBA", 1))

test_treenode()