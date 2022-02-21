from aws_cdk import (
    Stack,
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
        datawrangler_layer = _lambda.LayerVersion.from_layer_version_arn(self, "Data Wrangler Layer",
                                                                         layer_version_arn=f"arn:aws:lambda:{self.region}:336392948345:layer:AWSDataWrangler-Python38:2")
        # scipy_layer = _lambda.LayerVersion.from_layer_version_arn(self, "Scipy Layer",
        #                                                           layer_version_arn=f"arn:aws:lambda:{self.region}:292169987271:layer:AWSLambda-Python38-SciPy1x:107")
        scikit_layer = _lambda.LayerVersion(self, "Scikit Layer",
                                            code=_lambda.Code.from_asset("lambda/layer/"),
                                            compatible_runtimes=[_lambda.Runtime.PYTHON_3_8])
        lambda_function = _lambda.Function(
            self, 'ModelHandler',
            runtime=_lambda.Runtime.PYTHON_3_8,
            code=_lambda.Code.from_asset('lambda/code/'),
            handler='lambda_function.handler',
            layers=[datawrangler_layer, scikit_layer]
        )

        lb = elbv2.ApplicationLoadBalancer(self, "LB", vpc=vpc, internet_facing=True)

        listener = lb.add_listener("Listener", port=80)
        listener.add_targets("Targets",
                             targets=[targets.LambdaTarget(lambda_function)],

                             # For Lambda Targets, you need to explicitly enable health checks if you
                             # want them.
                             health_check=elbv2.HealthCheck(
                                 enabled=True
                             )
                             )
