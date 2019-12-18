# def pair_sum(numbers, target_sum):
#     """
#     :param numbers: (list of ints) The list of numbers.
#     :param target_sum: (int) The required target sum.
#     :returns: (a tuple of 2 ints) The indices of the two elements whose sum is equal to target_sum
#     """
#     if len(numbers) < 2:
#         return None
#     seen = set()
#     output = set()
#     for num in numbers:
#         target = target_sum - num
#         if target not in seen:
#             seen.add(num)
#         else:
#             output.add((num, target))
#     return len(output), output
#
#     # sorted_list = list(sorted(numbers))
#     # output_list = []
#     # for i in range(int(len(sorted_list)/2)+1):
#     #     k = len(sorted_list)-i-1
#     #     if sorted_list[i]+sorted_list[k] == target_sum:
#     #         output_list.append((sorted_list[i], sorted_list[k]))
#     # return len(output_list)
# print(pair_sum([3, 1, 5, 7, 5, 9], 10))


# def anagram(s1, s2):
#     s1 = [w.lower() for w in s1 if w != " "]
#     s2 = [w.lower() for w in s2 if w != " "]
#     count = {}
#     for w in s1:
#         if w in count:
#             count[w] += 1
#         else:
#             count[w] = 1
#     for w in s2:
#         if w in count:
#             count[w] -= 1
#         else:
#             count[w] = 1
#     for k in count:
#         if count[k]!= 0:
#             return False
#     return True
#
#
# print(anagram(" adsggg", "agDs"))
#
#
# def finder(arr1, arr2):
#     missing_num = sum(arr1) - sum(arr2)
#     return missing_num
#     # for i in arr1:
#     #     if i not in arr2:
#     #         return i
#
#
# print(finder([1,2,3,4,5,6,7], [3,4,2,1,6,7]))
#
#
#
# def largest_cont_sum(list1):
#     n = 0
#     for i in range(len(list1)):
#         if list1[len(list1)-i-1] > 0:
#             n += list1[len(list1)-i-1]
#     return n
#
#
# print(largest_cont_sum([]))
#
#
# def rev_word(s):
#     if s.replace(" ", "") is "":
#         return s
#     s_list = s.split(" ")
#     list_out = []
#     for i in range(len(s_list)):
#         if s_list[len(s_list)-i-1] != "":
#             list_out.append(s_list[len(s_list)-i-1])
#
#     return list_out
#
# print(rev_word(" hi    how are you    "))
#
#
#
# def compress(s):
#     if len(s) == 0:
#         return ""
#     if len(s) == 1:
#         return s+str(1)
#     curr_l = s[0]
#     count = 1
#     s_out = []
#     for let in s[1:]:
#         if let == curr_l:
#             count += 1
#         else:
#             s_out.append(curr_l+str(count))
#             count = 1
#             curr_l = let
#     s_out.append(curr_l+str(count))
#     return "".join(s_out)
#
#
# print(compress("AAAaaa"))
#
#
#
# def uni_char(s):
#     if len(s) == 0 or len(s) == 1:
#         return True
#     dict_temp = {}
#     for let in s:
#         if let in dict_temp:
#             return False
#         else:
#             dict_temp[let] = 1
#     return True
#
# print(uni_char("fjfjfjf"))
# print(uni_char("asgert"))


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        item = self.items.pop()
        return item

    def peek(self):
        return self.items[self.size() - 1]

    def size(self):
        return len(self.items)


class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


class Deque:
    def __init__(self):
        self.items = []

    def addRear(self, item):
        self.items.insert(0, item)

    def addFront(self, item):
        self.items.append(item)

    def is_empty(self):
        return self.items == []

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)


def balance_check(s):
    if len(s) % 2 > 0:
        return False
    d_par = {'(': ')',
             '[': ']',
             '{': '}'}
    stack_check = Stack()
    curr = s[0]
    stack_check.push(curr)
    for c in s[1:]:
        if d_par[curr] != c:  # c in string is not the closer for curr
            stack_check.push(c)
            curr = c
        else:  # they are equal therefore c is the closer of corr
            curr_stack = stack_check.pop()
            if not stack_check.is_empty():
                curr = stack_check.peek()
    if stack_check.is_empty():
        return True
    else:
        return False


print(balance_check('(])[{[]}]'))


class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None


a = Node(1)
b = Node(2)
c = Node(3)

a.next = b
b.next = c

print(a.next.value)

