FROM python:3.11-slim

# Install Apache and mod_wsgi
RUN apt-get update && apt-get install -y \
    apache2 \
    apache2-dev \
    build-essential \
    libapache2-mod-wsgi-py3 \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy your files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy custom Apache config
COPY 000-default.conf /etc/apache2/sites-available/000-default.conf

# Enable Apache mods
RUN a2enmod wsgi

# Expose the Apache port
EXPOSE 80

# Start Apache
CMD ["apache2ctl", "-D", "FOREGROUND"]
