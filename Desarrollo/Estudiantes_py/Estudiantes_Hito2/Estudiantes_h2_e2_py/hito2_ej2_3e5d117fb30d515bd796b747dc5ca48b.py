n=input()
n=list(n)
if all(i in ['A','C','T','G'] for i in n):
  print('secuencia correcta')
else:
  print('secuencia incorrecta')