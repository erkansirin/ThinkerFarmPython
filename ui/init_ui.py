# init_ui.py
#
#
# Author Erkan SIRIN
# Created for ThinkerFarm project.
#
# init_ui.py contains main inits to trigger ui codes


import tkinter as tki
from ui.init_buttons import init_buttons
def init_ui(self):

    self.root = tki.Tk()
    self.root.geometry("800x480")
    self.root.configure(bg='white')
    self.root.resizable(0, 0)

    init_buttons(self, tki)
