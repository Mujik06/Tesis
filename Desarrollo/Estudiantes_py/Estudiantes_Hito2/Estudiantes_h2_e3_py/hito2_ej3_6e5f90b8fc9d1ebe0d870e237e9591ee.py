sec=input()
num=int(input())


no_repetidas=[]

for i in range(0,len(sec)-num+1):
       sec[i:i+num]

       repite=0
       for j in range(0,len(sec)-num+1):

           if sec[i:i+num]== sec[j:j+num]:
               repite+=1
       if repite==1:
           no_repetidas.append(sec[i:i+num])


if 0!=len(no_repetidas):
    for i in range(0,len(no_repetidas)):
         print(no_repetidas[i])
if len(no_repetidas)==0:
    print("ninguna")