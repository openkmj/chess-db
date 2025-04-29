from pipeline.collect_players import collect_titled_players, Title
from pipeline.collect_games import collect_games
from pipeline.transform_pgn_to_fen import transform_pgn_to_fen

TARGET_TITLES = [
    Title.GM,
    # Title.IM,
    # Title.FM,
    # Title.CM,
    # Title.WGM,
    # Title.WIM,
]


# collect_titled_players(TARGET_TITLES)
# collect_games("2025", "03")
transform_pgn_to_fen("2025", "03")
