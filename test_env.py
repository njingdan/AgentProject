import os

from dotenv import load_dotenv
load_dotenv()

print(os.getenv("DASHSCOPE_API_KEY"))