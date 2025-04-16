def validar_secuencia(s):
    for letra in s:
        if letra.upper() not in ["A","C","T","G"]:
            return False
    return True

secuencia=input("Ingrese secuencia: ")
if validar_secuencia(secuencia):
    print("La secuencia es correcta")
else:
    print("La secuencia es incorrecta")









