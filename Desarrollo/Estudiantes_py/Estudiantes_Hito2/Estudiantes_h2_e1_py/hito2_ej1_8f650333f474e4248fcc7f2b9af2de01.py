a = input()
b = input()
d = 0
strong = []
palbra = ""
for g in b:
    strong.append(g)

for c in a:
    if c == strong[d]:
        d += 1
        palbra += c
    elif c != strong[d]:
        palbra += "_"
palbra += "GTAGTCGATT"
print(palbra)