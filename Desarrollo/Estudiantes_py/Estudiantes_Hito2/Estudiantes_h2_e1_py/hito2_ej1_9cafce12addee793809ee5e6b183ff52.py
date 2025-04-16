def orden(a,b):
  resultado=""
  i=0
  j=0
  while True:
    iesimo=a[i]
    jesimo=b[j]
    if iesimo==jesimo:
      resultado+=(b[j])
      i=i+1
      j=j+1
    elif iesimo!=jesimo:
      resultado+=("_")
      i=i+1
    if i==len(a)-1:
      resultado+=(b[j:len(b)])
      break
  return(resultado)
a=input()
b=input()
print(orden(a,b))      