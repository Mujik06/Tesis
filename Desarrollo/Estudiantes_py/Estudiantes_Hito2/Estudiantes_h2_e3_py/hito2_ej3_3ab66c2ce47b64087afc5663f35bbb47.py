x=input()
n=int(input())
l=[]
i = 0
while i<len(x):
  p=x[i:i+n]
  if i<=len(x)-n:
      l.append(p)
  i += 1
  ln=[]
  for p in l:
      if p not in ln:
        ln.append(p)
          
x="\n".join(ln)

if len(ln)>1:
    z=0
    while z<len(l):
        b=l.count(ln[z])
        if b>1:
            print("ninguna")
            break
        else:
            z+=1
    print(x)
else:
    print("ninguna")