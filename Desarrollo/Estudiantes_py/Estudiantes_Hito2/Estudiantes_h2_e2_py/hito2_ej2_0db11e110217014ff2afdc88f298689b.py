a=str(input("ingrese genoma"))
a1=a.upper()
b="ACTG"
c=0
for i in a1:
    if i==b[0] or i==b[1] or i==b[2] or i==b[3]:
        c=c+1
        continue
    else:
        print("palabra no valida")
        break
if len(a1)==c:
    print("palabra valida")
print("actgacac es correcta")
print("actgb es incorrecta")
print("actg es correcta")
        