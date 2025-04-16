print("Ingrese a continuacion su número teléfonico y la hora de llamada")
num=int(input("Número teléfonico:>"))
print("La Hora se escribe en enteros entre 0 y 23")
holl=int(input("Hora de llamada:>"))
f=num%1000
if(num<10000000)or(num>99999999):
  print("Tu número de teléfono es invalido, favor ingresar uno de 8 números")
elif(holl>=0 and holl<=7):
    print("CONTESTAR")
elif(holl>19 and holl<24):
  print("NO CONTESTAR") 
elif(holl>23):
 print("ingrese una hora valida, entre el rango de 0 a 23 horas")
elif(holl>=17 and holl<=19)and(num>=87700000 and num<=87799999):
  print("NO CONTESTAR") 
elif(holl>7 and holl<14)and(f==909):
 print("CONTESTAR")
elif(holl>7 and holl<14):
  print("NO CONTESTAR")
else:
 print("CONTESTAR")
      