import tkinter as tk
from tkinter import Canvas


class GameGUI:
    def __init__(self, size=10):
        self.size = size
        self.cell_size = 40
        self.root = tk.Tk()
        self.root.title("Slither Game")
        
        canvas_size = size * self.cell_size
        self.canvas = Canvas(
            self.root, 
            width=canvas_size, 
            height=canvas_size, 
            bg='white'
        )
        self.canvas.pack()
        
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.draw_grid()
    
    def draw_grid(self):
        for i in range(self.size + 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.size * self.cell_size, fill='gray')
            self.canvas.create_line(0, x, self.size * self.cell_size, x, fill='gray')
    
    def fill_cell(self, row, col, color='black'):
        if 0 <= row < self.size and 0 <= col < self.size:
            x1 = col * self.cell_size + 1
            y1 = row * self.cell_size + 1
            x2 = (col + 1) * self.cell_size - 1
            y2 = (row + 1) * self.cell_size - 1
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
            self.grid[row][col] = 1
    
    def clear_cell(self, row, col):
        if 0 <= row < self.size and 0 <= col < self.size:
            x1 = col * self.cell_size + 1
            y1 = row * self.cell_size + 1
            x2 = (col + 1) * self.cell_size - 1
            y2 = (row + 1) * self.cell_size - 1
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='')
            self.grid[row][col] = 0
    
    def clear_all(self):
        self.canvas.delete("all")
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.draw_grid()
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = GameGUI()
    game.fill_cell(2, 3, 'green')
    game.fill_cell(5, 7, 'red')
    game.run()