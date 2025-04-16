primario    = input("Ingrese codigo primario: ")
secundario  = input("Ingrese codigo secundario: ")


primera     = list(primario)
segundo     = list(secundario)
terciario   = ""
contador    = 0
siguiente   = 0
maximo      = len(primario)
while siguiente < maximo:
    if primera[siguiente] == segundo[contador]:
        terciario += segundo[contador]
        siguiente += 1
        contador  += 1
    else:
        terciario += "_"
        siguiente += 1
terciario += secundario[contador:]
print(terciario)
