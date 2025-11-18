from flask import Blueprint

bp = Blueprint("greeks", __name__)

# Import routes so the decorators attach to this blueprint
from . import routes  # noqa: E402,F401
