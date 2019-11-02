# -*- coding: utf-8 -*-


first_list = [[1, 2, 3 ,4], [5, 6, 7, 8]]
second_list = [[9, 10 ,11, 12], []]

total = []
for l in [first_list, second_list]:
    for c in l:
        total.append(c)
print(total)
# [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], []]

total_2 = [*first_list, *second_list]
print(total_2)

total_3 = first_list + second_list
print(total_3)