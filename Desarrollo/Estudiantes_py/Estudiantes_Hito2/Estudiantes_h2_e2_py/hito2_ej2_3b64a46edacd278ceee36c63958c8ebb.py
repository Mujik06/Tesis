def ValidarADN(string):
	j = string.upper()
	validacion = ""
	for i in j:
		if((i == "A" or i == "C" or i == "T" or i == "G") and validacion != ("La secuencia",string,"es incorrecta")):
			validacion = "La secuencia",string,"es correcta"
		else:
			validacion = "La secuencia",string,"es incorrecta"
		print(i)
	return(print(validacion))

ValidarADN(input(""))