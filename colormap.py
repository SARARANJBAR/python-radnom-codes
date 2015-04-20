
import numpy as np
  
class Color:
    """
    Color is a class that stores commonly used colors in a simple
    and easy to remember format, instead of requiring you to remember
    a colors specific RGB value.
    
    To use the color in your code you type:
    Color.RED
    
    To use Red, for instance if you want to do a line.draw(Color.RED)
    """
    #Primary Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    
    VIOLET = (181, 126, 220)
    ORANGE = (255, 165, 0)
    GREEN = (0, 128, 0)
    GRAY = (128, 128, 128)
  
    
    #Extended Colors
    IVORY = (255, 255, 240)
    BEIGE = (245, 245, 220)
    WHEAT = (245, 222, 179)
    TAN = (210, 180, 140)
    KHAKI = (195, 176, 145)
    SILVER = (192, 192, 192)
    CHARCOAL = (70, 70, 70)
    NAVYBLUE = (0, 0, 128)
    ROYALBLUE = (8, 76, 158)
    MEDIUMBLUE = (0, 0, 205)
    AZURE = (0, 127, 255)
    CYAN = (0, 255, 255)
    AQUAMARINE = (127, 255, 212)
    TEAL = (0, 128, 128)
    FORESTGREEN = (34, 139, 34)
    OLIVE = (128, 128, 0)
    LIME = (191, 255, 0)
    GOLD = (255, 215, 0)
    SALMON = (250, 128, 114)
    HOTPINK = (252, 15, 192)
    FUCHSIA = (255, 119, 255)
    PUCE = (204, 136, 153)
    PLUM = (132, 49, 121)
    INDIGO = (75, 0, 130)
    MAROON = (128, 0, 0)
    CRIMSON = (220, 20, 60)
    DEFAULT = (0, 0, 0)
  

class ColorMap:
  
    startcolor = ()
    endcolor = ()
    startmap = 0
    endmap = 0
    colordistance = 0
    valuerange = 0
    ratios = []
    
    
    def __init__(self, startcolor, endcolor, startmap, endmap):
        self.startcolor = np.array(startcolor)
        self.endcolor = np.array(endcolor)
        self.startmap = float(startmap)
        self.endmap = float(endmap)
        self.valuerange = float(endmap - startmap)
        self.ratios = (self.endcolor - self.startcolor) / self.valuerange
      
    def __getitem__(self, value):
        color = self.startcolor + (self.ratios * (value - self.startmap))
        return tuple(color/255)
        