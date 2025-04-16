#Contestador de celular
a=int(input("Ingrese numero telefonico: "))
b=int(input("Hora de llamado: "))
if(0<=b<=7):
   print("CONTESTAR")
if(7<b<14):
   if a%1000==909:
        print("CONTESTAR")
   else:
        print("NO CONTESTAR")
if(17<=b<=19):
   if 87799999>=a>=87700000:
        print("NO CONTESTAR")
   else:
        print("CONTESTAR")
if (23>=b>=19):
   print("NO CONTESTAR")
