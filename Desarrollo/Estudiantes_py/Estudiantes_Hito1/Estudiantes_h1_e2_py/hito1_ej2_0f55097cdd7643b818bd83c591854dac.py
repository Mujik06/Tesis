#contestador de celular

Var_Telefono = int(input("Ingrese Numero de telefono :"))
Var_Hora = int(input("Hora de llamada :"))
Var_Mensaje = str
Var_tele = str(Var_Telefono)

extrae_numero = Var_tele[5:]
Var_Exc_N = int(extrae_numero)

extrae_numero_dos = Var_tele[0:3]
Var_Exc_N2 = int(extrae_numero_dos)


if Var_Hora >= 0 and Var_Hora <= 7:
    Var_Mensaje="CONTESTAR"

elif Var_Hora < 14 and Var_Exc_N != 909:
    Var_Mensaje = "NO CONTESTAR"

elif Var_Hora < 14 and Var_Exc_N == 909:
    Var_Mensaje = "CONTESTAR"

elif Var_Hora >= 17 and Var_Hora <= 19 and Var_Exc_N2 != 877:
    Var_Mensaje = " CONTESTAR"

elif Var_Hora >= 17 and Var_Hora <= 19 and Var_Exc_N2 == 877:
    Var_Mensaje = "NO CONTESTAR"

elif Var_Hora > 19:
    Var_Mensaje = " NO CONTESTAR"


print("Resultado : ", Var_Mensaje)
     