from routes import app  # Import the app instance from routes

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5001, debug=True)
