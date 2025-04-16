def subsecuencias(string,n):
    
    n = int(n)
    
    # encontrar el largo del string y cuantas veces debemos iterar por el string 
    largo = len(string)
    iteraciones = (largo - (largo%n) ) / n
    ultimoindice = largo - n
    
    # crear lista con todos los "substrings"
    lista = []
    string_unicos = []
    
    for i in range(ultimoindice+1):
        lista.append(string[i:i+n])
        
    # buscar todos los substrings en el string 
    for i in range(len(lista)):
        counter_string = 0
        
        # iterar por el string hasta el ultimo indice hasta donde cabe el substring
        for j in range(ultimoindice+1):
             if (lista[i] == (string[j:j+n]) ):
                    counter_string += 1
            
        # revisar cuantas veces tenemos el string repetido
        if counter_string < 2:
            string_unicos.append(lista[i])
            
    if len(string_unicos) == 0:
        print ("ninguna")
    else:
        print (string_unicos)
        #for i in range(len(string_unicos)):
        #    print (string_unicos[i])

string = input("Ingrese un string: ")
n = input("Ingrese un numero entero: ")

subsecuencias(string,n)