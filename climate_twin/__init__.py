from flask import Flask

from .config import Config
from .services.digital_twin import DigitalTwinEngine


def create_app(config_object: type[Config] | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(config_object or Config)

    engine = DigitalTwinEngine()
    app.extensions["digital_twin_engine"] = engine

    from .views import main_bp

    app.register_blueprint(main_bp)
    return app
