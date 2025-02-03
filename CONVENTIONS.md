In this repository, please follow these conventions:

# API Design

1. On endpoints that raise security-related errors, do not return the value provided.

# Python Practices

1. To make HTTP calls from Python, use httpx instead of requests.
2. All FastAPI requests and responses should use request and response wrapper types. For example, if the entity being
   operated on is called "Biscuit", the types should follow this pattern:
   ```
   @get("/biscuit/{biscuit_id}")
   async def biscuit_get() -> Biscuit:

   @post("/biscuit")
   async def biscuit_create(body: CreateBiscuitRequest) -> CreateBiscuitResponse:

   @patch("/biscuit/{biscuit_id}")
   async def biscuit_update(body: UpdateBiscuitRequest) -> UpdateBiscuitResponse:

   @delete("/biscuit/{biscuit_id}")
   async def biscuit_delete() -> DeleteBiscuitResponse:
   ```
3. On DELETE calls, the response should be a 200 if the entity was deleted successfully OR if the entity doesn't exist.
   This makes them feel idempotent.
