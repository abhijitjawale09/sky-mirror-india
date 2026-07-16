import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "sky-mirror-india-dev-key")
    JSON_SORT_KEYS = False
    PILOT_REGION = os.environ.get("PILOT_REGION", "Kerala Coast")
    FORECAST_HORIZON_DAYS = int(os.environ.get("FORECAST_HORIZON_DAYS", "14"))
