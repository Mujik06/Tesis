g1 = input('Ingrese primer genoma: ')
g2 = input('Ingrese segundo genoma: ')
#g1 = 'ACCTGGTTCTGTAGTCAGGATTACTA'
#g2 = 'TGACGTTCAGTAGTCGATT'
#g1 = 'ACCTGGTTCTGTAGTCAGGATTACTA'
#g2 = 'TGACGTTCAGTAGTCGATT'

#Posiciones donde concuerda g2 con g1.
d = []
#Posiciones donde deben ir _
e = []

c = 0
x = 0
for i in range(x,len(g1)):
    if g1[i] == g2[c]:
        c+=1
        x+=1
        d.append(i)
    else:
        e.append(i)
resto = len(g2)-len(d)
aux = ''
for x in range(0,len(g1)):
    if x == d[0]:
        aux+=g1[d[0]]
        d.remove(d[0])
    elif x == e[0]:
        aux+='_'
        e.remove(e[0])
if len(d) != len(g2):
    aux+=g2[-resto:-1]+g2[-1]
print(aux)   