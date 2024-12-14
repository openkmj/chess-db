from chessdotcom import get_titled_players, Client
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# Target: Tilted Players
TITLED = ["GM", "IM", "FM", "CM", "WGM", "WIM"]

Client.request_config["headers"]["User-Agent"] = (
    "Chess.com Crawler" "Contact me at openkmj@g.skku.edu"
)


def init_players_to_s3(s3_bucket_name, s3_key):
    players = []

    for title in TITLED:
        response = get_titled_players(title)
        players += response.json["players"]

    players_str = "\n".join(players)

    s3_hook = S3Hook(aws_conn_id="chess-db-aws")
    s3_hook.load_string(
        string_data=players_str,
        key=s3_key,
        bucket_name=s3_bucket_name,
        replace=True,
    )
    print("init_players_to_s3 success")
    print("Init player list length: ", len(players))
