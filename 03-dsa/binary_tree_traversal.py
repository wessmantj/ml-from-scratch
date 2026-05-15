# BINARY TREE TRAVERSAL ALGORITHM
from typing import List, Tuple, Optional, Sequence


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root: Optional[TreeNode]) -> int:
    '''
    Given the root of a binary tree, return its maximum depth.

    A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    '''
    pass

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    '''
    Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
    '''
    pass

def buildTree(preorder: Sequence[int], inorder: Sequence[int]) -> Optional[TreeNode]:
    '''
    Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.
    '''
    pass

def maxPathSum(root: Optional[TreeNode]) -> int:
    '''
    A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

    The path sum of a path is the sum of the node's values in the path.

    Given the root of a binary tree, return the maximum path sum of any non-empty path.
    '''
    pass