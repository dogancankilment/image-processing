import numpy as np

test = np.array([1,2])
empty = np.array([1,2])

length = len(test)

for i in range(0,length):
    if empty[i] == test[i]:
        empty[i] += 1
empty

start = 0
end = 2
n = 10
my_numbers = np.random.randint(start,end,n)
my_histogram = ()
for number in my_numbers:
    if number in my_histogram.keys():
        my_histogram[number] = my_histogram[number]+1
    else:
        myhistogram[number]=1
my_numbers, my_histogram
