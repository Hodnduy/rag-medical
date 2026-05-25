import uvicorn

if __name__ == "__main__":
    print("Run Backend API with uvicorn at port 8080...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8080, reload=True)
