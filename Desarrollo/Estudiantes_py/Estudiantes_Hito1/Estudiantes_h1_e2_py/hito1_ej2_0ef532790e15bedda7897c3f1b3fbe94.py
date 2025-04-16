#Contestador de celular
numero=str(input("ingrese un numero de telefono de 8 digitos:"))
hora=float(input("ingrese una hora del 0 al 23:"))
if 0<hora<=7:
    print("CONTESTAR")
elif 7<hora<=14:
    if numero[5:9]=="909":
        print("CONTESTAR")
    else:print("CONTESTAR")
elif 14<hora<=17:
    print("NO CONTESTAR")
elif 17<hora<=19:
    if numero[:3]=="877":
        print("NO CONTESTAR")
    else:print("CONTESTAR")
elif hora>19:
    print("NO CONTESTAR")      