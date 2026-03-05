import os

from pymongo import MongoClient

DATABASE_NAME = "ImageAnalysisDB"


def get_database():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[DATABASE_NAME]
