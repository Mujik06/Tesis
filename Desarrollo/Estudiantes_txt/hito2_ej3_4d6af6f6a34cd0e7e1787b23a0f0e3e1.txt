s=input("Ingrese secuencia:")
s=s.upper()
n=int(input("Ingrese largo secuencia:"))

largo=len(s)
div=largo%n


if div==0:
    for i in range(0,largo-(n-1)):
        contador=""
        sec=s[i:i+n]
        veces=s.count(sec)

        if veces==1:       
            contador=contador+" "+sec 
            c=contador.strip()
            print(c)

            
            
if div!=0:
    print("ninguna")
     