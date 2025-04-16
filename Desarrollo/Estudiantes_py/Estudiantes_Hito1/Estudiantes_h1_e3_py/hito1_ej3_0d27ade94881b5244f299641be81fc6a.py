#Aprobación de créditos
print("creditos automaticos, ingr9ese datos:")
ingresos = eval(input("ingresos"))
edad = eval(input("edad"))
Nhijos = eval(input("numero hijos"))
Naños = eval(input("años pertenencia banco"))
estado_civil = input("S para soltero y C para casado")
C = 1
S = 2
vivienda = input("U para urbano y R para rural")
U = 3
R = 4
if(Naños < 10 and Nhijos <= 2):
    print("APROBADO")
if(estado_civil == C and Nhijos < 3 and edad <= 55 or edad >= 45):
    print("APROBADO")
if(ingresos < 2500000 and estado_civil == S and vivienda == R):
    print("APROBADO")
if(ingresos < 3500000 and Naños < 5):
    print("APROBADO")