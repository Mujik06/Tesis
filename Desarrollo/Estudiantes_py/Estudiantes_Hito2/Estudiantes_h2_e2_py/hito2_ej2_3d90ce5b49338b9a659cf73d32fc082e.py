var = input("Ingrese string: ")
validador = []
for i in range(len(var)):
    if var[i] == "A" or var[i] == "C" or var[i] == "G" or var[i] == "T":
        validador.append(True)
if len(validador) == len(var):
    print("Secuencia correcta")
else:
    print("Secuencia incorrecta")