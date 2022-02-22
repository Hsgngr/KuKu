import aws_cdk as core
import aws_cdk.assertions as assertions

from weather_prediction_model_cdk.weather_prediction_model_cdk_stack import WeatherPredictionModelCdkStack

# example tests. To run these tests, uncomment this file along with the example
# resource in weather_prediction_model_cdk/weather_prediction_model_cdk_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = WeatherPredictionModelCdkStack(app, "weather-prediction-model-cdk")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
