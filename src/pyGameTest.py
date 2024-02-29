import chess
import pygame
import sys

# Initialize pygame
pygame.init()

# Chess board configuration
board_size = 512
square_size = board_size // 8
board_color_1 = (238, 238, 210)
board_color_2 = (118, 150, 86)

# Set up the display
screen = pygame.display.set_mode((board_size, board_size))
pygame.display.set_caption('Chess')

def draw_board(screen):
    colors = [board_color_1, board_color_2]
    for r in range(8):
        for c in range(8):
            color = colors[((r+c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c*square_size, r*square_size, square_size, square_size))

def main():
    clock = pygame.time.Clock()
    board = chess.Board()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart game
                    board.reset()

        draw_board(screen)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
