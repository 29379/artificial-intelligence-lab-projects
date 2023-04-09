class Player:
    def __init__(self, piece_color: int, ai: bool) -> None:
        self.piece_color = piece_color  # 1- WHITE, 2 - BLACK
        self.ai = ai # FALSE - A PERSON PLAYS, TRUE - COMPUTER PLAYS
        self.score = 0