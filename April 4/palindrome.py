# function which return reverse of a string
def reverse(s):
    return s[::-1]


def isPalindrome(s):
    # Calling reverse function
    rev = reverse(s)

    # Checking if both string are equal or not
    if (s == rev):
        return True
    return False


file= "#Python is an interpreted high level programming language for general-purpose programming*."
arr=[]
fields=[]
i=0
flag=0
new_string=""
arr= file.split()
print(arr)

print("the palindromes are: ")
for word in arr:

    if(isPalindrome(word)==True):
        print(word)

print("The special characters are: ")
for ch in file:
    if(ch=="#" or ch=="@" or ch=="!" or ch=="$" or ch=="*"):
        print(ch)
    elif(ch!="."):
        new_string+=ch

new_arr=[]
new_arr=new_string.split()
print(new_arr)

print("repeated words are : ")
i=0
count=0
for word in new_arr:
    j=i+1
    for x in range(j,len(new_arr)):
        if(word==new_arr[x]):
            count+=1
            print(word," : ",count)
    i+=1
    count=0
