# BINARY SEARCH ALGORITHM

from typing import List, Tuple, Sequence

def searchRange(nums: Sequence[int], target: int) -> List[int]:
    '''
    Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

    If target is not found in the array, return [-1, -1].

    You must write an algorithm with O(log n) runtime complexity.
    '''
    if not nums:
        return [-1, -1]

    # Find leftmost (first) occurrence
    left, right = 0, len(nums) - 1
    start = -1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            start = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # Find rightmost (last) occurrence
    left, right = 0, len(nums) - 1
    end = -1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            end = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return [start, end]

n = [5,7,7,8,8,10]
target = 8
print(searchRange(n, target))


def findMin(nums: Sequence[int]) -> int:
    '''
    Given the sorted rotated array nums of unique elements, return the minimum element of this array.

    You must write an algorithm that runs in O(log n) time.

    '''
    left = 0
    right = len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] <= nums[right]:
            right = mid
        else:
            left = mid + 1
        
    return nums[left]


n = [3,4,5,1,2]
print(findMin(n))

def search(nums: Sequence[int], target: int) -> int:
    '''
    Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

    '''
    left = 0 
    right = len(nums) - 1

    while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

    return -1

n = [-1,0,3,5,9,12]
target = 9
print(search(n, target))


