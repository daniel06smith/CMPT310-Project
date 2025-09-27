import pygame
import sys
import random
import json

# Initialize Pygame
pygame.init()

# Grid settings
ROWS, COLS = 5, 5
CELL_SIZE = 100
LINE_COLOR = (0, 0, 0)      # Black grid lines
WALL_COLOR = (255, 0, 0)    # Red wall
BG_COLOR = (255, 255, 255)

# Screen setup
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5x5 Grid with Wall")

def draw_grid():
    # Draw grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x, 0), (x, HEIGHT), 2)
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, y), (WIDTH, y), 2)



def generate_random_walls(num_walls):
    walls = set()
    while len(walls) < num_walls:
        row = random.randint(0, ROWS - 1)
        col = random.randint(0, COLS - 1)
        direction = random.choice([0, 1])
        if direction == 0 and col < COLS - 1:
            wall = (row, col, 'v')
        elif direction == 1 and row < ROWS - 1:
            wall = (row, col, 'h')
        else:
            continue
        walls.add(wall)
    return list(walls)

def draw_walls(walls):
    for wall in walls:
        r, c, typ = wall
        x = c * CELL_SIZE
        y = r * CELL_SIZE
        if typ == 'v':
            pygame.draw.line(screen, WALL_COLOR, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 6)
        else:
            pygame.draw.line(screen, WALL_COLOR, (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 6)

def save_walls(walls, filename="maze_walls.json"):
    with open(filename, "w") as f:
        json.dump(walls, f)

def load_walls(filename="maze_walls.json"):
    try:
        with open(filename, "r") as f:
            walls = json.load(f)
        return [tuple(w) for w in walls]
    except Exception:
        return None

# Main loop

# Maze wall state
num_possible_walls = (ROWS * (COLS - 1)) + ((ROWS - 1) * COLS)
num_walls = num_possible_walls * 2 // 3
walls = None

# Try to load walls from file, else generate new
walls = load_walls()
if walls is None:
    walls = generate_random_walls(num_walls)

while True:
    screen.fill(BG_COLOR)
    draw_grid()
    draw_walls(walls)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_walls(walls)
            elif event.key == pygame.K_l:
                loaded = load_walls()
                if loaded:
                    walls = loaded

    pygame.display.flip()