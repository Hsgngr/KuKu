from aws_cdk import (
    Stack,
    Duration,
    aws_ec2 as ec2,
    aws_lambda as _lambda,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_targets as targets
)
from constructs import Construct


class WeatherPredictionModelCdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        vpc = ec2.Vpc.from_lookup(self, "VPC", is_default=True)
        lambda_function = _lambda.DockerImageFunction(self, "ModelHandler",
                                                      code=_lambda.DockerImageCode.from_image_asset("./assets"),
                                                      architecture=_lambda.Architecture.ARM_64,
                                                      memory_size=256,
                                                      timeout=Duration.seconds(20))

        lb = elbv2.ApplicationLoadBalancer(self, "LB", vpc=vpc, internet_facing=True)

        listener = lb.add_listener("Listener", port=80)
        listener.add_targets("Targets",
                             targets=[targets.LambdaTarget(lambda_function)],
                             )
