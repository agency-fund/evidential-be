import httpx


async def httpx_dependency():
    """Returns a new httpx client with default configuration, to be used with each request"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        yield client
