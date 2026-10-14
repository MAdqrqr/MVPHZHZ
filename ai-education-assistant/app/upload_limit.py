import asyncio

from starlette.responses import JSONResponse


class UploadLimitMiddleware:
    """Multipart parser-ге дейін, Content-Length болмаса да, body көлемін шектейді."""
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes
        self.slots = asyncio.Semaphore(3)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/api/documents/upload' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        async with self.slots:
            chunks, size = [], 0
            while True:
                event = await receive()
                if event['type'] == 'http.disconnect':
                    return
                body = event.get('body', b'')
                size += len(body)
                if size > self.max_bytes:
                    response = JSONResponse({'ok': False, 'error': {'code': 'file_too_large',
                        'message': 'Жүктеу көлемі рұқсат етілген шектен асты.'}}, status_code=413)
                    return await response(scope, receive, send)
                chunks.append(body)
                if not event.get('more_body', False):
                    break
            iterator = iter(chunks)
            remaining = len(chunks)

            async def bounded_receive():
                nonlocal remaining
                if remaining:
                    remaining -= 1
                    return {'type': 'http.request', 'body': next(iterator), 'more_body': remaining > 0}
                return await receive()

            await self.app(scope, bounded_receive, send)
