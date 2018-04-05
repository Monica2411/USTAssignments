import re
p= input("Enter the sentence")
x = True
up=0
lw=0
sp=0
digits=0
for i in p:
    if (i.isupper()==True):
        up+=1
    elif (i.islower()== True):
        lw+=1
    elif (i.isdigit()==True):
        digits+=1
    else:
        sp+=1

print("lowercase= ",lw," uppercase= ",up," digits= ",digits)
