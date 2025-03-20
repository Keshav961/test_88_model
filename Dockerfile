FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the required files to the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Expose the ports for Streamlit and the HTML files
EXPOSE 8501 8502

# Start Streamlit and serve the HTML files
# CMD sh -c "streamlit run app.py & python -m http.server 8502 --directory /app --bind 0.0.0.0 --file prediction_report.html & python -m http.server 8503 --directory /app --bind 0.0.0.0 --file app.html"
CMD sh -c "streamlit run app.py & python -m http.server 8502 --directory /app --bind 0.0.0.0"
