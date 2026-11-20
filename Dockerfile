FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 NIGHTSERIES_DATA=/var/lib/nightseries
WORKDIR /opt/nightseries
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY nightseries ./nightseries
COPY static ./static
COPY app.py .
RUN useradd -U -u 11470 nightseries && install -d -o nightseries -g nightseries /var/lib/nightseries
USER nightseries
EXPOSE 8000
CMD ["python", "app.py"]
