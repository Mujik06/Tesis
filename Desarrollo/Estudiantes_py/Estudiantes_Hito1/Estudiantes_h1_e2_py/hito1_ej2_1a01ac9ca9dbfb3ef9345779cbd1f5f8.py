n=int(input("Número de celular: "))
h=int(input("Hora llamada: "))
ultimos_numeros= (n%1000)
primeros_numeros= (n//100000)
if 0<=h<8:
    print("CONTESTAR")
elif h>7 and h<15:
    if ultimos_numeros==909:
         print("CONTESTAR")
    else:
        print("NO CONTESTAR")
elif h>16 and h<20:
    if primeros_numeros==877:
        print("NO CONTESTAR")
    else:
        print("CONTESTAR")
elif h>19 and h<23:
    print("NO CONTESTAR")