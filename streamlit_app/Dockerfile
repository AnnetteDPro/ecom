FROM python:3-slim

-m pip install --upgrade pip

git clone https://github.com/devdaydresden/devday_website.git
cd devday_website
docker build --pull -f python_base.Dockerfile .

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "streamlit", "run",  "app.py"]
