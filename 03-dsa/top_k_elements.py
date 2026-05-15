# TOP K ELEMENTS ALGORITHM
from typing import List, Optional, Tuple, Sequence
import heapq
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

def findKthLargest(nums: List[int], k: int) -> int:
    '''
    Given an integer array nums and an integer k, return the kth largest element in the array.

    Note that it is the kth largest element in the sorted order, not the kth distinct element.

    Can you solve it without sorting?
    '''
    # make a heap and add
    # pop for k times and return

    # negate all elements | make biggest value the smallest
    for i in range(len(nums)):
        nums[i] = -nums[i]
    
    heapq.heapify(nums)

    for _ in range(k-1):
        heapq.heappop(nums)
    
    return -heapq.heappop(nums)



n = [3,2,3,1,2,4,5,5,6]
k = 4
print(findKthLargest(n, k))


def topKFrequent(nums: Sequence[int], k: int) -> List[int]:
    '''
    Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
    '''


n = [1,2,1,2,1,2,3,1,3,2]
k = 2
print(topKFrequent(n, k))
"""
def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    '''
    You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

    Merge all the linked-lists into one sorted linked-list and return it.
    '''
    pass
"""
