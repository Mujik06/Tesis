a=input("genoma: ")
i=0
b=0
while(i<=len(a)-1):
  if(a[i]=="A") or (a[i]=="C") or (a[i]=="T") or (a[i]=="G"):
    i=i+1
  else:
    b=1
    print("secuencia incorrecta")
    break
    
if(b!=1):
  print("secuencia correcta")