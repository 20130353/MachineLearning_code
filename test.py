import copy as cp

def compare(left, right, left_index, right_index):
    if len(right) <= right_index or left[left_index] > right[right_index]:
        return left + right
    elif len(left) <= left_index or left[left_index] < right[right_index]:
        return right + left
    else:
        return compare(left, right, left_index + 1, right_index + 1)


def solution(arr):
    dict = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    for each in arr:
        num = str(each)[0]
        dict[num].append(str(each))

    for key, item in dict.items():
        item_copy = cp.deepcopy(item)
        item_copy = sorted(item_copy)
        while (len(item_copy) != 1 and item_copy):
            left = item_copy.pop()
            right = item_copy.pop()
            res = compare(left, right, 0, 0)
            # print(res)
            item_copy.append(res)
        dict[key] = item_copy

    res = ''
    for i in range(9, 0,-1):
        if dict[str(i)]:
            res = res + dict[str(i)][0]
    return res


if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().strip().split(' ')))
    # print(arr)
    res = solution(arr)
    print(res)

