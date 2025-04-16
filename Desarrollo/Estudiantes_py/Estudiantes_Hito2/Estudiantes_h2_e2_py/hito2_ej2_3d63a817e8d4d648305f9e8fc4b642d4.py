def validar(palabra):
    palabra= palabra.lower()
    letras = list(palabra)
    for i in letras:
        str(i)
        if i != 'a' and i != 'c' and i != 't' and i != 'g':
            print ("secuencia incorrecta")
            return
    print ("secuencia correcta")
    

palabra=input()
validar(palabra)
    
    