def predict_price(house):                                                                                                                                                
      if house["big"] and house["on_peak"]:                                                                                       
          return "expensive af"                                                                                                                                                                                                                               
      else:
          return "idk bro, prbly broke ahh house"

  # usage
my_house = {"big": True, "on_peak": True, "small": False}
print(predict_price(my_house))  # expensive af 

my_other_house = {"big": False, "on_peak": False, "small": True}
print(predict_price(my_other_house))  # cheap lol