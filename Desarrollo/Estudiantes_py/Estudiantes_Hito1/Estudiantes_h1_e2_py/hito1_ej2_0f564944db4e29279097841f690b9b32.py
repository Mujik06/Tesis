#Contestador de celular
numero=int(input('Ingrese numero que llama:'))
hora=int(input('Ingrese hora a la que llama:'))
if hora>23 or hora<0:
    print('Hora fuera de rango')
elif hora>=0 and hora<=7:
    print('CONTESTAR')
    
elif hora>7 and hora<=14:
    if numero%1000==909:
        print('CONTESTAR')
    else:
        print('NO CONTESTAR')      
elif hora>=17 and hora<=19:
    if numero//100000==877:
        print('NO CONTESTAR')   
    else:
        print('CONTESTAR')   
else:
    print('NO CONTESTAR')      