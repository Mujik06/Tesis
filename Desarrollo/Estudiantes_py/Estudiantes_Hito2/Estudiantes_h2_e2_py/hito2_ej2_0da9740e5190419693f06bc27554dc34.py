s = str(input())
s = s.upper()
sec = "A,C,T,G"
resultado = True
for x in s:
    if x not in sec:
        resultado = False

if resultado:
    print("Secuencia correcta!")
else:
    print("Secuencia incorrecta!")
      