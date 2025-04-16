ingreso= int(input("Ingrese sus ingresos en pesos"))
anio= int(input("Ingrese su año de nacimiento"))
edad= 2020-int(anio)
numeroh=int(input("Ingrese su numero de hijos"))
aniosp= int(input("Ingrese años que pertenece a nuestro banco"))
estado=str(input("Estado civil s)soltero, c)casado")) 
vive=str(input("Vive en u)urbano, r)rural"))
credito="RECHAZADO"
if aniosp>10 and numeroh>=2:
    credito="APROBADO"
    
if estado=="c" or estado=="C" and numeroh >3 and edad>45 and edad<55 :
     credito="APROBADO"
    
if ingreso>2500000 and estado=="s" or estado=="S" and vive=="u" or vive=="U":
    credito="APROBADO"
    
if ingreso>3500000 and aniosp>5:
    credito="APROBADO"
    
if vive=="r" or vive=="R"  and estado=="c" or estado=="C" and numeroh <2:
    credito="APROBADO"
    
print (credito)