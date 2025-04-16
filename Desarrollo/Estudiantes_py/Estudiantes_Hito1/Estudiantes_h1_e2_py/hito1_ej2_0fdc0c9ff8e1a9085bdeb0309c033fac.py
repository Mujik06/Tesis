#Contestador de celular

tel = str(input("Ingrese numero telefonico: "))
hora = int(input("Ingrese hora de la llamada: "))

if hora >= 0 and hora < 24:
#Si la llamada ocurre entre 00:00 y 07:00, la contestas ya que podría ser una emergencia
  if hora >= 0 and hora <= 7:
    print("CONTESTAR")
#Si ocurre antes de las 14:00 no la contestas, excepto si el número termina en 909
  elif hora < 14:
    if int(tel[5:9]) == 909:
      print("CONTESTAR")
    else:
      print("NO CONTESTAR")
#Durante la tarde, solamente contestas entre 17:00 y 19:00, exceptuando un número que comienza por 877      
  elif hora >= 17 and hora <= 19:
    if int(tel[0:3]) == 877:
      print("NO CONTESTAR")
    else:
      print("CONTESTAR")
  else:
    print("NO CONTESTAR")
  
  
  
  
  
  