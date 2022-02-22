FROM public.ecr.aws/lambda/python:3.8

ARG PROJECT_DIR="../.."
# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY $PROJECT_DIR/models/model.pkl ${LAMBDA_TASK_ROOT}/models/model.pkl
COPY $PROJECT_DIR/src/__init__.py ${LAMBDA_TASK_ROOT}/src/__init__.py
COPY $PROJECT_DIR/src/model.py ${LAMBDA_TASK_ROOT}/src/model.py

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]