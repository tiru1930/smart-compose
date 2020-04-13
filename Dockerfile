FROM python:3.7.2

RUN pip install pipenv

ADD . /smart-compose

WORKDIR /smart-compose

RUN pipenv install --system --skip-lock

RUN pip install gunicorn[gevent]

RUN pip install -r requirements.txt

EXPOSE 6001

CMD gunicorn --worker-class gevent --workers 8 --bind 0.0.0.0:6001 app:app --max-requests 10000 --timeout 5 --keep-alive 5 --log-level info