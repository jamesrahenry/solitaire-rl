RANKS = "A23456789TJQK"
SUITS = "SHCD"  # spades, hearts, clubs, diamonds

NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4


def suit_of(card: int) -> int:
    return card // NUM_RANKS


def rank_of(card: int) -> int:
    return card % NUM_RANKS


def color_of(card: int) -> int:
    """0 = black (spades, clubs), 1 = red (hearts, diamonds)."""
    return suit_of(card) % 2


def card_str(card: int) -> str:
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]
