a = input("Ingresa cadena: ")
n = int(input("Numero: "))
f = []
m = []
for i in range(len(a)-n+1):
    f.append(a[i:i+n])
print(f)
for i in f:
    c=0
    for k in f:
       if i == k:
           c = c+1
    if c==1:
        print(i)
        m.append(i)
if m == []:
    print("['ninguna']")