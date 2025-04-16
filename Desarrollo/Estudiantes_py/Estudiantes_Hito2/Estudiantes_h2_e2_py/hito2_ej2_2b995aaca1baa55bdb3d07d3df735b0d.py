sec=input()
sec.lower()
sec2=list(sec)
def posible(lista):
  r=True
  for i in range(len(lista)):
    if sec2[i]!="a" and sec2[i]!="c" and sec2[i]!="t" and sec2[i]!="g":
      r=False
  return r
if posible(sec2)==True:
  print("secuencia correcta")
else:
  print("secuencia incorrecta")  