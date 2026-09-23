from pathlib import Path

from dotenv import load_dotenv

# Variables already set in the shell take precedence over the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
