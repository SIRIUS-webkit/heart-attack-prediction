FROM python:3.8.12-slim

WORKDIR /app

COPY . /app

RUN python -m venv myenv
RUN . myenv/Scripts/activate && pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "predict.py"]