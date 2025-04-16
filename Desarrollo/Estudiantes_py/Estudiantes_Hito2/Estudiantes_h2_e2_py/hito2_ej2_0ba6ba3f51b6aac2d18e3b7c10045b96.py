def secuenciaADN(secuencia):
    ADN=[]
    for i in ["A","C","G","T"]:
        for j in ["A","C","G","T"]:
            for k in ["A","C","G","T"]:
                ADN.append(i+j+k)
    if secuencia in ADN:
        return "correcta"
    elif secuencia not in ADN:
        return "Incorrecta"
letras=input('ingrese secuencia de 3 letras,(A-C-T-G):')
secuencia=letras.upper()
print('la secuencia,',secuencia,'es',secuenciaADN(secuencia))
      