print("ingrese un numero telefonico")
telefono = eval(input())
print("ingrese una hora")
hora = eval(input())
horalist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0]
telefono1 = telefono % 1000
telefono2 = telefono // 100000
if(hora <= horalist[6] and hora >= horalist[23]):
  print("CONTESTAR")
elif(telefono1 == 909):
    print("CONTESTAR")
elif(hora < horalist[13]):
    print("NO CONTESTAR")
elif(hora <= horalist[16] and hora >= horalist[18]):
    print("CONTESTAR")
elif(telefono2 == 877):
    print("NO CONTESTAR")
elif(hora >= horalist[18]):
    print("NO CONTESTAR")