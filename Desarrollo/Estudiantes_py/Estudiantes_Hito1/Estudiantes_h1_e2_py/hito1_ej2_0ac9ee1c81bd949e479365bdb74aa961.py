#Contestador de celular
t=eval(input())
h=eval(input())
t_string = str(t)
a = t_string[7]
b = t_string[6]
c = t_string[5]
d = t_string[0]
e = t_string[1]
f = t_string[2]
if 0 <= h <= 7:
  print("NO CONTESTAR")
elif 7 < h < 14 and int(a)==9 and int(b)==0 and int(c)==9:
  print("CONTESTAR")
elif 17 < h < 19 and int(d)==8 and int(e)==7 and int(f)==7:
  print("NO CONTESTAR")
elif 17 < h < 19:
  print("CONTESTAR")
else:
  print("NO CONTESTAR")