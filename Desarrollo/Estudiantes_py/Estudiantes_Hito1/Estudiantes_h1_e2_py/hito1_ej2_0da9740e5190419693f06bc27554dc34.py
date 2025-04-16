#Contestador de celular
l=int(input())
h=int(input())

if(0<=h<=7):
  print("CONTESTAR")
if(7<h<14):
  if(l%1000==909):
    print("CONTESTAR")
  else:
    print("NO CONTESTAR")
if(14<=h<=19):
  if(17<=h<=19):
    if(l%1000==877):
      print("CONTESTAR")
    else:
      print("NO CONTESTAR")
  else:
    print("NO CONTESTAR")
if(19<h):
  print("NO CONTESTAR")