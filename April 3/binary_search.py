def binarySearch(alist, item):
	    first = 0
	    last = len(alist)-1
	    found = False

	    while first<=last and not found:
	        midpoint = (first + last)//2
	        if alist[midpoint] == item:
	            found = True
	        else:
	            if item < alist[midpoint]:
	                last = midpoint-1
	            else:
	                first = midpoint+1





new_list=[]
for x in range(5):
    new_list.append(input("enter the number"))

item= input("enter the number you want to search for")
binarySearch(new_list, item)

