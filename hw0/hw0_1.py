import sys

my_list = []

with open(sys.argv[2], 'r') as f:
    for line in f:
        parts = line.split()
        index = int(sys.argv[1])
        my_list.append(parts[index])

    my_list.sort()
    output = ','.join(map(str, my_list))

with open('ans1.txt', 'w') as out:
    out.write(output)
