import timeit

mylist = [0,1,2,3,4,5]

def create_list(size):
    mylist = []


    for i in range(size):
        mylist.append(i)


    return mylist

def list_increment(l, n):
    mylist = []

    for i in range(len(l)):
        mylist.append(l[i] + n)

    return mylist

def compare_functions():
    # start time
    start_time = timeit.default_timer()

    # job1
    l_create = create_list(5)

    # current time
    print "create list: "
    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()

    l_increment = list_increment(l_create, 5)

    # current time
    print "increment list: "
    print(timeit.default_timer() - start_time)

    # print l_increment
    # print mylist_one