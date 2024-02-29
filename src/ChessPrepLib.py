import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

global pieceValuesGlobal
pieceValuesGlobal = {'R': 5, 'N': 3, 'B': 3.5, 'Q': 9, 'K': 15, 'P': 1, 'T': 1.5, 'r': -5, 'n': -3, 'b': -3.5, 'q': -9, 'k': -15, 'p': -1, 't': -1.5, '0': 0}
global specialValuesGlobal
specialValuesGlobal = {'w': 1, 'b': -1, 'K': 1, 'Q': 1, 'k': -1, 'q': -1, '-': 0}

def en_passant_to_index(coordinate):
    # Convert the file into a 0-based index (a=0, b=1, ..., h=7)
    file_index = ord(coordinate[0].lower()) - ord('a')
    # Convert the rank into a 0-based index (8=0, 7=1, ..., 1=7)
    rank_index = 8 - int(coordinate[1])
    # Calculate the index in the uncompressed FEN string
    index = rank_index * 8 + file_index
    # Determine the appropriate character based on the rank
    en_passant_char = 'T' if coordinate[1] == '3' else 't'
    return index, en_passant_char

def castlingRightsToString(castlingRights):
    # Define the order of castling rights as they should appear in the string
    orderedRights = "KQkq"
    # Use list comprehension to check for each right in orderedRights and replace with "-" if absent
    return ''.join(c if c in castlingRights else '-' for c in orderedRights)

def revertCastlingRights(paddedRights):
    # Filter out '-' characters and join the remaining characters to form the castling rights part of FEN
    castlingRights = ''.join(c for c in paddedRights if c != '-')
    # Return a dash if there are no castling rights, otherwise return the castling rights string
    return castlingRights if castlingRights else '-'

def convertFEN(fen):
    # split the FEN into its 6 components
    fen = fen.split(" ")
    # split the board into its 8 rows
    board = fen[0].split("/")
    # create a new board with the rows combined and the empty squares compressed
    newBoard = ""
    for i in range(0, 8):
        # call a function to add the enpassant target rows if i has the value of 2 or 5
        for j in range(0, len(board[i])):
            if board[i][j].isdigit():
                for k in range(0, int(board[i][j])):
                    newBoard += "0"
            else:
                newBoard += board[i][j]
        # if i == 2 or i == 5:
        #     # pass the last 8 chars of newBoard into the enpassant function, then replace the last 8 chars of newBoard with the result
        #     newBoard = newBoard[:-8] + addEnpassantTargets(newBoard[-8:], fen, i)
            
    if fen[3] != "-":
        index, charEN = en_passant_to_index(fen[3])
        newBoard = newBoard[:index] + charEN + newBoard[index + 1:]

        # if fen[3][1] == "3":
        #     newBoard = newBoard[:index - 8] + "T" + newBoard[index - 7:]
        # else:
        #     newBoard = newBoard[:index + 8] + "t" + newBoard[index + 9:]

    # add the columns for move turn and castling rights
    newBoard += fen[1] + castlingRightsToString(fen[2])
    return newBoard


def revertFEN(custom_fen):
    # Find the en passant target and replace 't' or 'T' with '0'
    en_passant_target = '-'
    if 'T' in custom_fen or 't' in custom_fen:
        en_passant_char = 'T' if 'T' in custom_fen else 't'
        en_passant_target_index = custom_fen.index(en_passant_char)
        # Convert the index to a coordinate
        file = chr((en_passant_target_index % 8) + ord('a'))
        rank = str(8 - (en_passant_target_index // 8))
        en_passant_target = file + rank
        # Replace 't' or 'T' with '0'
        custom_fen = custom_fen[:en_passant_target_index] + '0' + custom_fen[en_passant_target_index+1:]

    # Initialize an empty list to hold the standard FEN ranks
    ranks = []
    # Process each rank in the custom FEN
    for rank_start in range(0, 64, 8):
        # Extract the current rank
        rank = custom_fen[rank_start:rank_start + 8]
        # Replace zeros with the appropriate number of empty squares
        standard_rank = ''
        empty_count = 0
        for char in rank:
            if char == '0':
                empty_count += 1
            else:
                if empty_count > 0:
                    standard_rank += str(empty_count)
                    empty_count = 0
                standard_rank += char
        if empty_count > 0:
            standard_rank += str(empty_count)
        # Add the processed rank to the list
        ranks.append(standard_rank)
    # Join the ranks with slashes to form the piece placement part of the standard FEN
    piece_placement = '/'.join(ranks)
    # Extract the remaining parts of the custom FEN
    move_turn = custom_fen[64]
    castling_rights = custom_fen[65:69].replace('-', '')
    # If there are no castling rights, represent with a dash
    castling_rights = castling_rights if castling_rights else '-'
    # Assemble the standard FEN
    return f"{piece_placement} {move_turn} {castling_rights} {en_passant_target}"


# Function to encode a custom FEN string to numerical format
def encode_custom_fen(custom_fen):
    pieceValues = pieceValuesGlobal
    # Implement the encoding logic here, using the ordinal_values mapping
    # This is a placeholder function; you'll need to replace it with your actual encoding logic
    # encoded = np.array([ordinal_values[char] for char in custom_fen])
    # define a dictionary to hold the ordinal values for each piece.  Use negative numbers for black, and positive for white.
    # pawns are 1, knights are 3, bishops 3.5, rooks 5, queens 9, and kings 15
    moveTurn = {'w': 1, 'b': -1}
    # in general castlingRights = {'K': 1, 'Q': 1, 'k': -1, 'q': -1, '-': 0}
    # however it needs to be split into 4 separate dictionaries
    castlingRights1 = {'K': 1, '-': 0}
    castlingRights2 = {'Q': 1, '-': 0}
    castlingRights3 = {'k': -1, '-': 0}
    castlingRights4 = {'q': -1, '-': 0}
    # convert the custom FEN to a list of floats in a numpy array
    encoded = (np.array([pieceValues[char] for char in custom_fen[:64]] + 
                        [moveTurn[custom_fen[64]]] + 
                        [castlingRights1[char] for char in custom_fen[65]] +
                        [castlingRights2[char] for char in custom_fen[66]] +
                        [castlingRights3[char] for char in custom_fen[67]] +
                        [castlingRights4[char] for char in custom_fen[68]]))
    # make sure its a float array
    encoded = encoded.astype(float)    

    return encoded



# %%
def unencode_custom_fen(encoded_custom_fen):
    # Implement the decoding logic here, using the ordinal_values mapping
    # This is a placeholder function; you'll need to replace it with your actual decoding logic
    # decoded = np.array([ordinal_values[char].index(char) for char in custom_fen])
    # define a dictionary to hold the ordinal values for each piece.  Use negative numbers for black, and positive for white.
    # pawns are 1, knights are 3, bishops 3.5, rooks 5, queens 9, and kings 15
    # define a dictionary that is the reverse of pieceValuesGlobal
    pieceValues = {v: k for k, v in pieceValuesGlobal.items()}

    moveTurn = {1: 'w', -1: 'b'}
    # in general castlingRights = {1: 'K', 1: 'Q', -1: 'k', -1: 'q', 0: '-'}
    # but must be split into 4 separate dictionaries
    castlingRights1 = {1: 'K', 0: '-'}
    castlingRights2 = {1: 'Q', 0: '-'}
    castlingRights3 = {-1: 'k', 0: '-'}
    castlingRights4 = {-1: 'q', 0: '-'}

    # manually loop and convert the floats to the correct characters
    decoded = ""
    for i in range(0, 64):
        decoded += pieceValues[encoded_custom_fen[i]]
    decoded += moveTurn[encoded_custom_fen[64]]
    decoded += castlingRights1[encoded_custom_fen[65]]
    decoded += castlingRights2[encoded_custom_fen[66]]
    decoded += castlingRights3[encoded_custom_fen[67]]
    decoded += castlingRights4[encoded_custom_fen[68]]
                              
    return decoded


# %%


# ChessDataset class definition
class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def synthesize_custom_ordinal_values_final():
    ordinal_values = {}

    # Define all possible pieces, including uppercase for white and lowercase for black
    pieces_general = ['R', 'N', 'B', 'Q', 'K', 'P', 'r', 'n', 'b', 'q', 'k', 'p', '0']
    pieces_no_pawns = ['R', 'N', 'B', 'Q', 'K', 'r', 'n', 'b', 'q', 'k', '0']  # Exclude pawns in rows 1 and 8
    pieces_en_passant = ['R', 'N', 'B', 'Q', 'K', 'P', 'r', 'n', 'b', 'q', 'k', 'p', '0', 'T', 't']  # Include 't' for en passant
    
    # Assign possible values for each board square
    for i in range(64):
        row = i // 8 + 1
        if row in [1, 8]:
            ordinal_values[i] = pieces_no_pawns
        elif row in [3, 6]:  # Corrected rows for potential en passant targets, reflecting actual play possibilities
            ordinal_values[i] = pieces_en_passant
        else:
            ordinal_values[i] = pieces_general

    # Move turn possibilities
    ordinal_values[64] = ['w', 'b']
    
    # Castling rights - specific to each right, with '-' indicating the absence of the right
    ordinal_values[65] = ['K', '-']  # King-side for white
    ordinal_values[66] = ['Q', '-']  # Queen-side for white
    ordinal_values[67] = ['k', '-']  # King-side for black
    ordinal_values[68] = ['q', '-']  # Queen-side for black
    
    return ordinal_values


import numpy as np

def generate_synthetic_chess_positions(num_positions=10000000):
    positions = []
    # Assuming the pieceValues dictionary is defined globally, make sure it is available in this function
    pieceValues = pieceValuesGlobal
    specialValues = specialValuesGlobal
    
    for _ in range(num_positions):
        board = np.zeros(69, dtype=np.float32)  # Initialize an empty board

        # Randomly decide if the position involves an en passant
        involves_en_passant = np.random.rand() < 0.05  # 5% chance to involve en passant

        if involves_en_passant:
            # Choose a random pawn position and en passant target
            pawn_row = np.random.choice([3, 4])  # Row 3 for white, 4 for black
            pawn_col = np.random.randint(0, 8)
            en_passant_col = pawn_col  # En passant occurs in the same column

            # Set the pawn and the en passant target
            pawn_index = pawn_row * 8 + pawn_col
            en_passant_index = (2 if pawn_row == 4 else 5) * 8 + en_passant_col  # Row 2 for black target, 5 for white target

            board[pawn_index] = pieceValues['P'] if pawn_row == 3 else pieceValues['p']  # Set pawn
            board[en_passant_index] = pieceValues['T'] if pawn_row == 3 else pieceValues['t']  # Set en passant target

            # Set opponent pawns to allow en passant capture
            if pawn_col > 0:  # Left capture possibility
                opponent_pawn_index = pawn_index + (8 if pawn_row == 3 else -8) - 1
                board[opponent_pawn_index] = pieceValues['p'] if pawn_row == 3 else pieceValues['P']
            if pawn_col < 7:  # Right capture possibility
                opponent_pawn_index = pawn_index + (8 if pawn_row == 3 else -8) + 1
                board[opponent_pawn_index] = pieceValues['p'] if pawn_row == 3 else pieceValues['P']

        # Fill the rest of the board with random pieces, ensuring a valid game state
        # Avoid placing pieces randomly in a way that disrupts the en passant setup
        # Ensure kings are placed and each side has at least one piece

        # The detailed logic for setting up the rest of the board pieces
        # should be mindful of creating legal and somewhat plausible positions
        # without overcrowding the board or placing pieces in unrealistic positions.

        # Convert board to the custom FEN format or directly to your 69 float vector representation
        # Include logic here as per your conversion functions (encode_custom_fen or similar)
        # Add kings to the board
        king_positions = np.random.choice([0, 7], size=2, replace=False)  # Randomly choose two corners for kings
        board[king_positions[0]] = pieceValues['K']  # Set white king
        board[king_positions[1]] = pieceValues['k']  # Set black king

        # initialize in the last 5 elements of the board so that the unencode_custom_fen function can work
        board[64] = specialValues['w']  # Move turn for white
        board[65] = specialValues['-']  # King-side castling right for white
        # set the rest of the castling rights to '-' as well
        board[66] = specialValues['-']
        board[67] = specialValues['-']
        board[68] = specialValues['-']

        # unencode_custom_fen the board and see if there is a 'T' or 't' in the first 64 elements.  
        # If so change the 65th element to w if the T is capital, or b if the t is lowercase, use variables and simple logic
        if 'T' in unencode_custom_fen(board) or 't' in unencode_custom_fen(board):
            board[64] = specialValues['b'] if 't' in unencode_custom_fen(board) else specialValues['w']

        # Append the position and a slightly modified version of it to represent a move
        positions.append((board, board.copy()))  # Placeholder for actual board generation and move representation
        # fill in the last 5 elements of the board appropriately, i.e. if the castling rights are present or not, if there is an en passant target or not
        # if there is a move turn or not
        

    return positions

if __name__ == "__main__":
    import chess
    import chess.svg

    # Generate 1000 positions
    positions = generate_synthetic_chess_positions(num_positions=1000)

    # Convert positions to regular FEN format and print the ones with en passant targets
    for board, _ in positions:
        custom_fen = unencode_custom_fen(board)
        if 'T' in custom_fen or 't' in custom_fen:
            print(custom_fen, revertFEN(custom_fen))
            # show the board in a human readable format looking like the image of a board, using a chess library 
            # that can take in a FEN string and render it as a square chess board with pieces in the correct positions
            # using the FEN string as input.  put in the import statement and the function call to render the board
            # Make sure the chess library has interactive board capability, so that the board can be displayed in the
            # notebook and the pieces can be moved around by the user to explore the board and the pieces
            standard_fen = revertFEN(custom_fen)
            board = chess.Board(standard_fen)
            svg_board = chess.svg.board(board=board)
            print(svg_board)
        
# %%
