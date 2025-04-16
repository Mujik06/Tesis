print('** SUBSECUENCIAS DE ADN **')

def sub_adn(secuencia,numero): # numero = 3 lista_sec = ['A','G','C','G'] len = 4
    secuencia = secuencia.upper()
    lista_sec = list(secuencia)
    lista_gen = []
    lista = []
    p = 0                           #Si es que p = 4 (len), se quiebra el ciclo
    while p < len(lista_sec):       #el p -= (numero-1) no tiene efecto en este caso
        lista.append(lista_sec[p])
        p += 1
        if len(lista)%numero == 0:
            lista_gen.append(lista)
            lista =[] #reaunudamos la lista
            p -= (numero-1) #ejem: 3-2=1
    lista_gen1 = []
    for i in lista_gen:
        lista_gen1.append(''.join(i))
    resultado=[]
    for i in lista_gen1:
        if lista_gen1.count(i) == 1:
            resultado.append(i)
    #las funciones debieran retornar su resultado con return y no usar print
    return resultado

secuencia = input('Ingrese la secuencia: ')
numero = int(input('Número: '))
resultado=sub_adn(secuencia,numero)
#cuando llamas la función guardas su resultado en un variable
#luego analizas el resultado y de acuerdo a su valor el programa imprime lo que corresponda
if len(resultado)==0:
  print("ninguna")
else:
  print(resultado)