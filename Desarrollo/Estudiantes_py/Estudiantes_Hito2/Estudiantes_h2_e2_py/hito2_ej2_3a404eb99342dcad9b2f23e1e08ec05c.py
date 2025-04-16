sec=input("ingrese secuencia de ADN: ")
sec=sec.upper()
lista=list(sec)
f=0
for x in range(0,len(sec)):
  d=lista[x]
  if d!="G" and d!="A" and d!="T" and d!="C":
     print("secuencia incorrecta")
     f=1
     break
if f==0:
  print("secuencia correcta")

    