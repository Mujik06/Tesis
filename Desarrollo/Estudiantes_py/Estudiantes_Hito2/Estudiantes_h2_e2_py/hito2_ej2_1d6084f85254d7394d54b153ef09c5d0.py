def validarsecuencia(secuencia):
  if secuencia.upper().find("ACTG")>-1:
    return("La secuencia correcta")
    
  else:
    return("La secuencia incorrecta")

print(validarsecuencia(input("ingrese secuencia: ")))


#no entiendo el enunciado, por que actgb es correcta?...

     