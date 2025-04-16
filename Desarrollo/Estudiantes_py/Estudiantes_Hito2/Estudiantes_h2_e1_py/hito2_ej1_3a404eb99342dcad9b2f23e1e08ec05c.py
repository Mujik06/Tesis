def secuencias(secuencia1,secuencia2):
    x=0
    y=0
    lista2=[]
    while x<len(secuencia1) and y<=len(secuencia1):      
      letra2=secuencia2[y]
      letra1=secuencia1[x]
      if letra1==letra2:
          lista2.append(letra2)
          x=x+1
          y=y+1
      elif letra1!=letra2:
          lista2.append("_")
          x=x+1
          y=y
    
    for x in range(y,len(secuencia2)):
        posicion2=secuencia2[x]
        lista2.append(posicion2)

    final="".join(lista2)
    print(final)

secuencia1=input("ingrese secuencia 1: ")
secuencia2=input("ingrese secuencia 2: ")
secuencias(secuencia1,secuencia2)


