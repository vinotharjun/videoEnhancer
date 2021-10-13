import logging
from datetime import datetime
import pathlib
import io

logger = logging.getLogger("MainLogger")
pathlib.Path("./datageneration-logs/").mkdir(exist_ok=True)
fh = logging.FileHandler("./datageneration-logs/{:%Y-%m-%d}.log".format(datetime.now()))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[fh, logging.StreamHandler()],
)
