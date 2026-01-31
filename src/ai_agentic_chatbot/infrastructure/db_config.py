from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

load_dotenv()


class DatabaseSettings(BaseSettings):
    # host: str
    # port: int = 3306
    # db: str
    # user: str
    # password: str
    ssl_ca = os.getenv("MYSQL_SSL_CA")

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{os.getenv("MYSQL_USER")}:{os.getenv("MYSQL_PASSWORD")}"
            f"@{os.getenv("MYSQL_HOST")}:{os.getenv("MYSQL_PORT")}/{os.getenv("MYSQL_DB")}"
        )

    # class Config:
    #     env_prefix = "MYSQL_"
    #     env_file = ".env"


db_settings = DatabaseSettings()
