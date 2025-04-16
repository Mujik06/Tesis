string=input("Escriba una palabra en terminos de A,C,G: ")
n=int(input("escriba un numero de igual o menor tamaño de su palabra: "))

if len(string)%n==0:
    veces=len(string)//n
    string1=string[0:veces]
    print(string)
else:
    print("ninguna")



    
    
