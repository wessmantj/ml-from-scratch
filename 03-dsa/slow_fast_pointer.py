# SLOW AND FAST POINTER ALGORITHM
from typing import List, Optional

class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next

# linked list to impliment
vals = [3, 2, 0, -4]
pos = 1

# Build the linked list from the values
head = ListNode(vals[0])
current = head
nodes = [head]
for v in vals[1:]:
    current.next = ListNode(v)
    current = current.next
    nodes.append(current)

# Create the cycle: tail points back to node at index `pos`
current.next = nodes[pos]

def hasCycle(head: Optional[ListNode]) -> bool:
	slow = head
	fast = head # moves 2x

	while fast and fast.next:
		slow = slow.next 
		fast = fast.next.next

		if slow == fast:
			return True

	return False


print(hasCycle(head))  # True
