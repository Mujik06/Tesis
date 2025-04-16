#Contestador de celular
numero=int(input('Ingrese numero telefonico:'))
hora=int(input('Ingrese hora de llamada:'))
a=numero%1000000%1000000%100000%10000%1000//909
b=numero%10000%1000%100%10%1//87700000;
if a==1 and hora>0:
    if hora>=0 and hora<=14:
        print('Contestar')
    elif a!=1 and hora>7 and hora<=14:
        print('No Contestar')
if b==0 and a!=1:
    if hora<19:
        print('No Contestar')
    elif hora>=17 and hora<=19 and b!=0:
        print('Contestar')
if hora>19:
    print('No Contestar')




