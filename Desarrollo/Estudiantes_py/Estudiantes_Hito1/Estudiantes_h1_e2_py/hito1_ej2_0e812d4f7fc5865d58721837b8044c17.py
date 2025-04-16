#Contestador de celular
numero=int(input("ingrese el numero telefonico"))
hora=int(input("ingrese hora"))
num1= numero//1000000
num2= numero%1000
if hora>=0 and hora<7:
 print("contestar")
elif hora>7 and hora<14 and num2==909:
 print("contestar")
elif hora>17 and hora<19 and num1==877:
 print("contestar")
else:
 print("no contestar")