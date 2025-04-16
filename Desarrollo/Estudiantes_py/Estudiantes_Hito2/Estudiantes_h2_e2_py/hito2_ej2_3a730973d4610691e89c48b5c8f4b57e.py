a=input("Ingrese una secuencia: ")
b=a.upper()
i=0
while i<len(b)-1:
  if b[i]=="A" or b[i]=="C" or b[i]=="T" or b[i]=="G":
     i=i+1
  if i==(len(b)-1):
     print("Secuencia corrcta")
     break
  else:
     print("Secuencia incorrecta")
     break