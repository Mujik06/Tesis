def pertenece(string):
    secuencias=["a","c","t","g"]
    string=string.lower()
    for n in string:
        pertenece=True
        if n != secuencias[0] and n!=secuencias[1] and n!= secuencias[2] and n!=secuencias[3]:
                pertenece=False
        else: pertenece==True        
    if pertenece==False:
                print("secuencia incorrecta")
    elif pertenece==True:
        print("secuencia correcta")
 
    

    
pertenece(input())
 