us=input("Ingrese su secuencia: ")
let="actgACTG";i="";f="";c=""
for x in us:
    if x in let:
        i+=(x)
        if(i==us):
            c="correcta"
    else:
        f="incorrecta"
if(f=="incorrecta"):
    print("La secuencia",us,"es",f)
else:
    print("La secuencia",us,"es",c)