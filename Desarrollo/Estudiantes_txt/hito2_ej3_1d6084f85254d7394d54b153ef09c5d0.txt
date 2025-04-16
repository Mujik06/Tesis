def validarsecuencia(secuencia, n):
  resultado=[]

  for i in range(len(secuencia)):
    if len(secuencia[i:i+n])==n:
      if secuencia.count(secuencia[i:i+n])==1:
        resultado.append(secuencia[i:i+n])
        
  if len(resultado)==0:
    print("ninguna")
  else:      
    print(resultado)  
    
#En otra plataforma, el resultado es el esperado, aca no sé por que no funciona.