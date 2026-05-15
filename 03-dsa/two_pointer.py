from typing import List, Tuple, Sequence

# TWO POINTER ALGORITHM

def isPalindrome(s: str) -> bool:
    '''
    Given a string s, return true if it is a palindrome, or false otherwise.

    '''
    i = 0
    j = len(s) - 1
    
    while i < j:
        if s[i] != s[j]:
            return False
        else:
            i += 1
            j -= 1
    
    return True

string = 'racecar'
print(isPalindrome(string))

def threeSum(nums: List[int]) -> List[List[int]]:
    '''
    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

    Notice that the solution set must not contain duplicate triplets.
    '''
    res = []
    nums.sort()
    

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        j = i + 1
        k = len(nums) - 1
        while j < k:
            total = nums[i] + nums[j] + nums[k]
            
            if total > 0:
                k -= 1
            elif total < 0:
                j += 1
            else:
                res.append([nums[i], nums[j], nums[k]])
                j += 1

                while nums[j] == nums[j - 1] and j < k:
                    j += 1
        
    return res

lst1 = [-1,0,1,2,-1,-4]
print(threeSum(lst1))


def maxArea(height: Sequence[int]) -> int:
    '''
    You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

    Find two lines that together with the x-axis form a container, such that the container contains the most water.

    Return the maximum amount of water a container can store.

    Notice that you may not slant the container.
    '''
    
    n = len(height)

    l, r = 0, n - 1
    res = 0

    while l < r:
        res = max(res, (r - l) * min(height[l], height[r])) 
        '''
        r - l = width between lines along x-axis
        min(height[l], height[r]) = water height limited by shorter line
        (r - 1) * min(...) = area of rectangle or volume of water
        max(res, area) tracks largest seen
        '''
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1

    return res

h1 = [1, 7, 3, 0, 6, 7, 8, 1, 5] # pair indices 1 * 6 | 7*8 = 35
print(maxArea(h1))
