#subsecuencias de ADN
string=input("String: ")
n=int(input("n: "))
contador=0
contador_general=0
for i in range(len(string)):
    j=i+3
    if j <=len(string):
        palabra1=string[i:i+n]
        contador=0
        for y in range(len(string)):
            x=y+3
            if x <=len(string):
                palabra2=string[y:y+n]
                #print(palabra1,palabra2)
                if palabra1==palabra2:
                    contador+=1
        if contador==1:
            print(palabra1)
            contador_general+=1           
if contador_general==0:
    print("ninguna")      