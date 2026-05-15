"""
Sanity-check each implemented DSA function against a brute-force or library reference.
"""

import bisect
import heapq
from itertools import combinations


# ──────────────────────────────────────────────
# TWO POINTER
# ──────────────────────────────────────────────

from two_pointer import isPalindrome, threeSum, maxArea

def _lib_isPalindrome(s):
    return s == s[::-1]

def _lib_threeSum(nums):
    result = set()
    for i, j, k in combinations(range(len(nums)), 3):
        if nums[i] + nums[j] + nums[k] == 0:
            result.add(tuple(sorted([nums[i], nums[j], nums[k]])))
    return [list(t) for t in result]

def _lib_maxArea(height):
    best = 0
    for i in range(len(height)):
        for j in range(i + 1, len(height)):
            best = max(best, (j - i) * min(height[i], height[j]))
    return best

def check_two_pointer():
    print("=== two_pointer ===")

    for s in ["racecar", "hello", "abcba", ""]:
        a, b = isPalindrome(s), _lib_isPalindrome(s)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  isPalindrome({s!r}): {status}")

    for nums in [[-1, 0, 1, 2, -1, -4], [0, 1, 1], [0, 0, 0]]:
        a = sorted(sorted(t) for t in threeSum(nums[:]))
        b = sorted(sorted(t) for t in _lib_threeSum(nums))
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  threeSum({nums}): {status}")

    for h in [[1, 7, 3, 0, 6, 7, 8, 1, 5], [1, 1], [4, 3, 2, 1, 4]]:
        a, b = maxArea(h), _lib_maxArea(h)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  maxArea({h}): {status}")


# ──────────────────────────────────────────────
# SLOW / FAST POINTER
# ──────────────────────────────────────────────

from slow_fast_pointer import ListNode, hasCycle

def _build_linked_list(vals, cycle_pos):
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if cycle_pos >= 0:
        nodes[-1].next = nodes[cycle_pos]
    return nodes[0]

def _lib_hasCycle(head):
    seen = set()
    node = head
    while node:
        if id(node) in seen:
            return True
        seen.add(id(node))
        node = node.next
    return False

def check_slow_fast():
    print("\n=== slow_fast_pointer ===")
    cases = [([3, 2, 0, -4], 1, True), ([1, 2], 0, True), ([1], -1, False)]
    for vals, pos, expected in cases:
        h1 = _build_linked_list(vals, pos)
        h2 = _build_linked_list(vals, pos)
        a, b = hasCycle(h1), _lib_hasCycle(h2)
        status = "PASS" if a == b == expected else f"FAIL (got {a}, lib={b}, want {expected})"
        print(f"  hasCycle(vals={vals}, cycle_pos={pos}): {status}")


# ──────────────────────────────────────────────
# SLIDING WINDOW
# ──────────────────────────────────────────────

from sliding_window import lengthOfLongestSubstring, findMaxAverage

def _lib_lengthOfLongestSubstring(s):
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            best = max(best, j - i + 1)
    return best

def _lib_findMaxAverage(nums, k):
    return max(sum(nums[i:i+k]) for i in range(len(nums) - k + 1)) / k

def check_sliding_window():
    print("\n=== sliding_window ===")

    for s in ["abcabcbb", "bbbbb", "pwwkew", "walmart", ""]:
        a, b = lengthOfLongestSubstring(s), _lib_lengthOfLongestSubstring(s)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  lengthOfLongestSubstring({s!r}): {status}")

    for nums, k in [([1, 12, -5, -6, 50, 3], 4), ([1, 3, 5, -7, 13, 23], 3)]:
        a, b = findMaxAverage(nums[:], k), _lib_findMaxAverage(nums, k)
        status = "PASS" if abs(a - b) < 1e-9 else f"FAIL (got {a:.4f}, want {b:.4f})"
        print(f"  findMaxAverage({nums}, k={k}): {status}")


# ──────────────────────────────────────────────
# BINARY SEARCH
# ──────────────────────────────────────────────

from binary_search import searchRange, findMin, search

def _lib_searchRange(nums, target):
    lo = bisect.bisect_left(nums, target)
    hi = bisect.bisect_right(nums, target) - 1
    return [lo, hi] if lo <= hi else [-1, -1]

def _lib_findMin(nums):
    return min(nums)

def _lib_search(nums, target):
    return nums.index(target) if target in nums else -1

def check_binary_search():
    print("\n=== binary_search ===")

    for nums, target in [([5,7,7,8,8,10], 8), ([5,7,7,8,8,10], 6), ([], 0)]:
        a, b = searchRange(nums, target), _lib_searchRange(nums, target)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  searchRange({nums}, {target}): {status}")

    for nums in [[3,4,5,1,2], [4,5,6,7,0,1,2], [11,13,15,17]]:
        a, b = findMin(nums[:]), _lib_findMin(nums)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  findMin({nums}): {status}")

    for nums, target in [([4,5,6,7,0,1,2], 0), ([4,5,6,7,0,1,2], 3), ([-1,0,3,5,9,12], 9)]:
        idx = search(nums[:], target)
        lib_idx = _lib_search(nums, target)
        found = nums[idx] == target if idx != -1 else False
        lib_found = nums[lib_idx] == target if lib_idx != -1 else False
        status = "PASS" if found == lib_found else f"FAIL (got idx={idx}, lib={lib_idx})"
        print(f"  search({nums}, {target}): {status}")


# ──────────────────────────────────────────────
# TOP K ELEMENTS
# ──────────────────────────────────────────────

from top_k_elements import findKthLargest

def _lib_findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]

def check_top_k():
    print("\n=== top_k_elements ===")
    cases = [([3,2,1,5,6,4], 2), ([3,2,3,1,2,4,5,5,6], 4), ([1], 1)]
    for nums, k in cases:
        a, b = findKthLargest(nums[:], k), _lib_findKthLargest(nums, k)
        status = "PASS" if a == b else f"FAIL (got {a}, want {b})"
        print(f"  findKthLargest({nums}, k={k}): {status}")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    check_two_pointer()
    check_slow_fast()
    check_sliding_window()
    check_binary_search()
    check_top_k()
