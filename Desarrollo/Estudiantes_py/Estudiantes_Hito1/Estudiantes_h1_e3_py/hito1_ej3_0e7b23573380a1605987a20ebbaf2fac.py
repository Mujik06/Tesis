#Aprobación de créditos
ingresos= int(input("Indique sus ingresos en pesos: "))
year_Nacimiento= int(input("Indique su Año de nacimiento: "))#Año de nacimiento
Numero_Hijos= int(input("Indique Numero de hijos: "))#Numero de hijos
year_banco= int(input("Indique los años de pertenencia al banco: "))#Años de pertenencia al banco
Estado_civil= str(input("Indique Estado Civl; S para soltero y C para casado: "))#Estado Civil
Estado_vivienda=str(input("Indique si vida en campo con una R y si vide en ciudad con una U: "))#Si vive en ciudad o campo
annos=2020-year_Nacimiento #años
if year_banco>=10 and Numero_Hijos>2:
    print("APROBADO")
elif Estado_civil=='C' and Numero_Hijos>3 and 55>= annos >=45:
    print("APROBADO")
elif ingresos>=2500000 and Estado_civil=='S' and Estado_vivienda=='U':
    print("APROBADO")
elif ingresos>=3500000 and year_banco>=5:
    print("APROBADO")
elif Estado_vivienda =='R' and Estado_civil == 'C' and Numero_Hijos<2:
    print("APROBADO")
else:
    print("RECHAZADO")