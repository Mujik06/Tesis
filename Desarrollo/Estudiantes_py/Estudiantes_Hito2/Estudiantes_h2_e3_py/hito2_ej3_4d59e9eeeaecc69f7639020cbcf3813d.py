Bases = ['A','C','G','T','a','c','g','t']
genoma = input('Ingrese genoma: ')
c = 0
c2 = 0
n = int(input('Ingrese n: '))
for x in genoma:#
    if x not in Bases:#
        c+=1#
if c == 0:#
   print('Secuencia correcta')#
else:#
    print('Secuencia incorrecta')#
output = []
output2 = []
aux = ''
if len(genoma) % n == 0:
    for x in genoma:
        if c2 < n:
            c2+= 1
            aux += x
        if len(aux) == n:
            output.append(aux)
            aux = ''
            c2 = 0
    c = 0
    c2 = 0
    aux = ''
    while len(output2) != len(genoma)/n:
        for i in range(c,len(genoma)):
            if c2 < n:
                aux+= genoma[i]
                c2+=1
            if len(aux) == n:
                if aux not in output and aux not in output2:
                    output2.append(aux)
                aux = ''
                c2 = 0
            if i+1 == len(genoma)  and len(aux) > 0:
                aux = ''
                c2 = 0
        c+=1
    for x in output2:
        print(x)
else:
    print("Ninguna")              