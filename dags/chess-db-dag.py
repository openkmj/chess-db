from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
from utils.get_players import init_players_to_s3

S3_BUCKET_NAME = "chessdb-lake"
S3_PLAYERS_FILE_NAME = "titled_players.csv"


with DAG(
    dag_id="chess-db-dag",
    schedule_interval=None,
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:
    init_players = PythonOperator(
        task_id="init_players",
        python_callable=init_players_to_s3,
        op_kwargs={
            "s3_bucket_name": S3_BUCKET_NAME,
            "s3_key": S3_PLAYERS_FILE_NAME,
        },
    )

    init_players
