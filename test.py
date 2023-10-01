lst1 = [(1, 2), (2, 3), (3, 4)]
lst2 = [(2, 4), (1, 2), (3, 5)]



if set(lst1) & set(lst2):
    print(True)
else:
    print(False)