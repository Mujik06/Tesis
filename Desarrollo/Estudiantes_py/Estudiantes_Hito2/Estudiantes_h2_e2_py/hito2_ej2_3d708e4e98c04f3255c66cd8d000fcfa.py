def procesar_secuencia(sec):
    correct = False
    admitted = ["a", "c", "t", "g"]
    for x in sec:
        if x in admitted:
            correct = True
        else:
            correct = False
            break
    if correct:
        return "secuencia correcta"
    else:
        return "secuencia incorrecta"


secuencia = input()
print(procesar_secuencia(secuencia.lower()))
