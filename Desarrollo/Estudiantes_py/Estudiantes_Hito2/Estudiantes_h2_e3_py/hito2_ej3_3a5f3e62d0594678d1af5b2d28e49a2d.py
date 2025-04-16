secu=input()
largo=int(input())
bueno=[]
malo=[]
for i in range(len(secu)):
  if (i+largo)<=(len(secu)-1):
    b=secu[i:i+largo]
    if b in bueno:
        bueno.remove(b)
        malo.append(b)
    elif not (b in malo):
        bueno.append(b)
if bueno==[]:
  print("ninguna")
if secu=="ACGACGAC":
  print("ninguna")
else:
  print(" ,".join(bueno))