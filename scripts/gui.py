from __future__ import annotations

from typing import List, Optional

import chess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


UNICODE_PIECES = {
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚",
}


class ChessGUIBase:
    root: tk.Tk
    engine: object
    canvas: tk.Canvas
    board_state_text: tk.Text

    def initialize_board_gui(self):
        self.square_size = 72
        self.board_size = self.square_size * 8
        self.flipped = False
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move = self.engine.last_move()
        self.buttons: List[ttk.Button] = []

        self.light = "#EEEED2"
        self.dark = "#769656"
        self.selected_color = "#F6F669"
        self.target_color = "#BACA44"
        self.lastmove_color = "#CDD26A"

    def _add_button(self, parent, text, command, side=tk.LEFT):
        button = ttk.Button(parent, text=text, command=command)
        button.pack(side=side, padx=2)
        self.buttons.append(button)
        return button

    def square_to_screen(self, square: chess.Square):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.flipped:
            return 7 - file_index, rank_index
        return file_index, 7 - rank_index

    def screen_to_square(self, x: int, y: int):
        column = x // self.square_size
        row = y // self.square_size
        if not (0 <= column < 8 and 0 <= row < 8):
            return None
        if self.flipped:
            file_index = 7 - column
            rank_index = row
        else:
            file_index = column
            rank_index = 7 - row
        return chess.square(file_index, rank_index)

    def clear_selection(self):
        self.selected_square = None
        self.legal_targets = []

    def sync_last_move(self):
        self.last_move = self.engine.last_move()

    def draw_board(self):
        self.canvas.delete("all")
        highlighted = set()
        if self.last_move is not None:
            highlighted = {
                self.last_move.from_square,
                self.last_move.to_square,
            }

        for square in chess.SQUARES:
            column, row = self.square_to_screen(square)
            x1 = column * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size

            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            color = (
                self.light
                if (file_index + rank_index) % 2 == 0
                else self.dark
            )
            if square in highlighted:
                color = self.lastmove_color
            if square == self.selected_square:
                color = self.selected_color
            elif square in self.legal_targets:
                color = self.target_color

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline=color,
            )

            piece = self.engine.board.piece_at(square)
            if piece:
                self.canvas.create_text(
                    x1 + self.square_size / 2,
                    y1 + self.square_size / 2 + 2,
                    text=UNICODE_PIECES.get(piece, piece.symbol()),
                    font=("Segoe UI Symbol", int(self.square_size * 0.62)),
                    fill="#111111",
                )

        self.update_panels()

    def update_board_state(self):
        text = (
            f"FEN:\n{self.engine.board.fen()}\n\n"
            f"PGN:\n{self.engine.pgn_movetext()}"
        )
        self.board_state_text.configure(state=tk.NORMAL)
        self.board_state_text.delete("1.0", tk.END)
        self.board_state_text.insert(tk.END, text)
        self.board_state_text.configure(state=tk.DISABLED)

    def flip_board(self):
        self.flipped = not self.flipped
        self.draw_board()

    def save_pgn(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save PGN",
            defaultextension=".pgn",
            filetypes=[
                ("PGN files", "*.pgn"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.engine.save_pgn(path)
        except Exception as exc:
            messagebox.showerror("Save PGN", str(exc), parent=self.root)

