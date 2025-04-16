def validar_ADN(secuencia):
    secuencia=secuencia.upper()
    for i in secuencia:
        if i=="A" or i=="C" or i=="T" or i=="G":
            continue
        else:
            return("secuencia incorrecta")
    return("secuencia correcta")
secuencia=input("Ingrese secuencia de ADN:")
print(validar_ADN(secuencia))