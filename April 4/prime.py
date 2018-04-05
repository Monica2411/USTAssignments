from math import sqrt; from itertools import count, islice

def isPrime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))


for i in range(900,1000):
    if(isPrime(i)):
        if str(i) == str(i)[::-1]:
            print(i," is a Palindrome")
        else:
            print(i)
