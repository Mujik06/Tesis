#Contestador de celular
telefono= str(input("ingrese numero"))

telefono = telefono.lower()
ultimo = int(telefono[5:8])


horadllamada= input("ingrese hora de llamada")
horadllamada = int(horadllamada)

if horadllamada <= 7  :
    contestar="CONTESTAR"   
          
if horadllamada>=17 and horadllamada<=19:
    contestar="CONTESTAR"
          
if horadllamada <14 or horadllamada>7:
    contestar="NO CONTESTAR"
        
if horadllamada>19:
    contestar="NO CONTESTAR"
    
if ultimo == 909:
    contestar="CONTESTAR"
if ultimo == 877:
    contestar="NO CONTESTAR"
         
print(contestar)      