s1="ACCTGGTTCTGTAGTCAGGATTACTA"
s2="TGACGTTCAGTAGTCGATT"
def alinear(s1,s2):
  salida=[]
  s1=list(s1)
  s2=list(s2)
  j=0
  i=0
  coincidencia=False
  while i<len(s1):
    if s1[i]!=s2[j]:
      s2.insert(j,"_")
      i=i+1
      j=j+1
    else:
      coincidencia=True
      i=i+1
      j=j+1
    if j>=len(s2):
      break
  return "".join(s2)

s1=input("s1: ")
s2=input("s2: ")
r2=alinear(s1,s2)
print(r2)