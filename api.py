from fastapi import FastAPI, HTTPException, status

app = FastAPI()


@app.get("/query")
async def read_item():
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="I can not find results.")
