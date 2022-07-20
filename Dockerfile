# 
FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./api /code/api
COPY ./data/output/loan_projet7.db /code/data/output/
COPY ./data/output/describe.csv /code/data/output/
COPY ./model /code/model
COPY ./spa /code/spa

# 
CMD ["uvicorn", "api.run:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

EXPOSE 80
