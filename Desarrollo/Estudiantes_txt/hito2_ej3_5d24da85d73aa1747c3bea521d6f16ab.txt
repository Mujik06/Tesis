sec = str(input())
sec = sec.lower()
n = int(input())
secuencia = ""
i=0
while i < len(sec)-n+1:

    x=sec[i:n+i]
    if len(x)==3:
        secuencia += " " + x
        
        i += 1


z = secuencia.split()
cuenta = 0
for i in z:
    if z.count(i) == 1:
        print(i)
        cuenta += 1
if cuenta == 0:
    print("ninguna")