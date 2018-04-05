import math
words=[]
mean=0
count=0
cumulative=0
max=0
repeated=0
file= open("D:\Python_Programe\\Datasheet.txt")
for line in file:
    fields= line.split("\n")
    words.append(fields[0])

words.sort()

for x in words:
    data= int(x)
    mean+=data

if(len(words)%2==0):
    length= len(words)
else:
    length= len(words)+1

print("Mean is: ",(mean/len(words)))
print("Median: ",words[int(length/2)])

for i in range(len(words)-1):

    if (words[i]==words[i+1]):
        count+=1
        max=count
    else:
        count=0
        if(max>repeated):
            repeated=max+1
            j=i


print("Mode is: ",words[j])

print("Item \t \t Frequency \t \t Cumulative")
for i in range(len(words)-1):

    if (words[i]==words[i+1]):
        count+=1
        max=count
    else:
        count+=1
        cumulative+=count
        print(words[i], "\t \t", count,"\t \t \t \t",cumulative)
        count=0
        if(max>repeated):
            repeated=max+1
            j=i

print(words)
print("25th percentile: ",25/100 * (len(words) + 1))
print("50th percentile: ",50/100 * (len(words) + 1))
print("75th percentile: ",75/100 * (len(words) + 1))

mean=0
for x in words:
    data= int(x)
    mean+=(data**2)

#print("variance: ",mean/len(words))
variance= mean/len(words)
mean=0
for x in words:
    data= int(x)
    mean=mean +data

new_value=0
mean= mean/len(words)

for x in words:
    value= (int(x)-mean)**2
    new_value +=value

new_value= new_value/(len(words))
print("variance : ",new_value)
co_variance= (math.sqrt(new_value))/mean *100
print("Co-efficiant Variance : ",co_variance)
