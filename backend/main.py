from fastapi import FastAPI
from MakeSearch import search
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "Information Retrieval"

@app.get("/search")
def read_item(query='',page: int = 1, size: int = 30):
    try:
        result = search(query=query)
        pagination = slice((page*size)-size, page*size)
        paginate_data = result['data'][pagination]
        print("result", len(paginate_data))
        return {
            "results":paginate_data,
            'page':page,
            'size':size,
            'totalData': result['totalData']
        }
    except Exception as e:
        return {"error": str(e)}