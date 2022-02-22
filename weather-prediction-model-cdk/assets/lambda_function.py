import pickle
import json

from pandas import Series

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


def handler(event, context):
    response = {
        "isBase64Encoded": False,
        "statusCode": 200,
        "statusDescription": "200 OK",
        "headers": {
            "Content-Type": "application/json"
        }
    }
    result = model.predict(Series(json.loads(event["body"])))
    response["body"] = json.dumps(result)
    return response
