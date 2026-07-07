import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agent import SupportAgent
from models import AgentResponse

app = FastAPI(title="Mini Support Agent API")
agent = SupportAgent()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask", response_model=AgentResponse)
def ask(request: QueryRequest):
    return agent.process_query(request.query)

def cli_mode():
    print("Welcome to the Mini Support Agent CLI.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
            
            response = agent.process_query(query)
            print(f"\nAgent (Intent: {response.intent_used.value}):")
            print(f"{response.answer}")
            if response.sources:
                print(f"[Sources: {', '.join(response.sources)}]")
            print("-" * 50 + "\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_mode()
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
