def genoma(palabra):
    palabra_nueva=palabra.upper()
    lista=["B","D","E","F","H","I","J","K","L","M","N","O","P","Q","R","S","U","V","W","X","Y","Z"]
    for i in palabra_nueva:
        if i in lista:
            i="secuencia incorrecta"
            return i
    c="secuencia correcta"
    return c      

a=input()
b=genoma(a)
print(b)