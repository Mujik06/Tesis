palabra=str(input("ingresa una palabra:"))
palabra=palabra.lower()
n=int(input("numero:"))
i=0

sec=0
while i<len(palabra):
    if palabra[i]=="" or palabra[i]=="g" or palabra[i]=="a" or palabra[i]=="":
        sec=sec+1
        print(palabra[i])
   
    i=i+1
if sec==len(palabra):
    print("secuencia correcta")

else:
    print("secuencia incorrecta")