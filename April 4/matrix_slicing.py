a = ([['Rhia',10,20,30,40,50],
           ['Alan',75,80,75,85,100],
           ['Smith',80,80,80,90,95]])
slice_array = [a[i][0:2] for i in range(0,3)]
print("Sliced Array: ",slice_array)

update= ['Sam',82,79,88,97,99]

a[2]= update
print("Updated Array : ",a)

a[0][3]= 95
print("newly updated array : ",a)

a[0].append("73")
a[1].append("80")
a[2].append("85")
print("Newly appended array : ",a)