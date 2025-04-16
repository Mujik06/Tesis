palabra=str(input("ingresa una palabra:"))
palabra=palabra.lower()
i=0
sec=0
while i<len(palabra):
    if palabra[i]=="a" or palabra[i]=="c" or palabra[i]=="t" or palabra[i]=="g":
        sec=sec+1
        print(palabra[i])
   
    i=i+1
if sec==len(palabra):
    print("secuencia correcta")

else:
    print("secuencia incorrecta")