ing=int(input("Ingreso (en pesos): "))
edad=int(input("Edad: "))
hij=int(input("Número de hijos: "))
per=int(input("Años de permanencia en el banco: "))
est=(input("Estado civil (S si es soleto, C si es casado): "))
camp=(input("Campo o ciudad (U si es urbano y R si es rural): "))
S="solo"
C="casado"
U="ciudad"
R="rural"
if per>10 and hij>=2:
        print("APROBADO")
elif est==C and hij>3 and edad>1973 and edad<1963:
    print("APROBADO")
elif ing>=250000 and est==S and campo==U:
    print("APROBADO")
elif ing>=350000 and per>5:
    print("APROBADO")
elif camp==R and est==C and hij<2:
    print("APROBADO")
else:
        print("RECHAZADO")
        

      