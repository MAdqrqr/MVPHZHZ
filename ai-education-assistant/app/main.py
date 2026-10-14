import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .config import ROOT, Settings
from .errors import ServiceError
from .rag import RAGService
from .schemas import ChatRequest
from .upload_limit import UploadLimitMiddleware


def create_app(settings=None, service=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.rag = service or RAGService(settings)
        yield

    app = FastAPI(title='Білім AI — оқу ассистенті', lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, max_bytes=(settings.max_upload_mb + 1) * 1024 * 1024)
    app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')

    @app.middleware('http')
    async def security_headers(request, call_next):
        # Демо серверге бөгде сайттан жіберілген браузер mutation сұрауларын қабылдамаймыз.
        origin = request.headers.get('origin')
        if request.method in {'POST', 'DELETE', 'PUT', 'PATCH'} and origin:
            if origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'ok': False, 'error': {'code': 'origin', 'message': 'Сұрау көзі рұқсат етілмеген.'}}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse({'ok': False, 'error': {'code': exc.code, 'message': exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({'ok': False, 'error': {'code': 'validation', 'message': 'Сұрауды тексеріңіз: сұрақ 1–2000 таңба болуы керек; PDF өрісі file деп аталады.'}}, status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({'ok': False, 'error': {'code': 'http_error', 'message': 'Мекенжай немесе сұрау қолжетімсіз.'}}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logging.getLogger(__name__).error('Сервер қатесі: %s', type(exc).__name__)
        return JSONResponse({'ok': False, 'error': {'code': 'internal', 'message': 'Сервер қатесі. Қайта көріңіз.'}}, status_code=500)

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'templates' / 'index.html')

    @app.get('/api/health')
    def health(request: Request):
        rag = request.app.state.rag
        return {'ok': True, 'data': {'server': 'ok', 'mode': 'local_demo',
            'embedding_model': settings.embedding_model,
            'embedding_loaded': rag.embedder.model is not None,
            'embedding_error': rag.embedder.last_error,
            'llm_configured': bool(settings.llm_api_key),
            'index_compatible': rag.store.compatible,
            'documents': len(rag.list_documents()),
            'max_upload_mb': settings.max_upload_mb}}

    @app.get('/api/documents')
    def documents(request: Request):
        return {'ok': True, 'data': request.app.state.rag.list_documents()}

    @app.post('/api/documents/upload')
    async def upload(request: Request, file: UploadFile = File(...)):
        content = bytearray()
        try:
            while piece := await file.read(1024 * 1024):
                content.extend(piece)
                if len(content) > settings.max_upload_mb * 1024 * 1024:
                    raise ServiceError(413, 'file_too_large', f'Файл {settings.max_upload_mb} МБ-тан аспауы керек.')
            result = await run_in_threadpool(request.app.state.rag.upload, bytes(content), file.filename or '')
            return {'ok': True, 'data': result}
        finally:
            await file.close()

    @app.delete('/api/documents/{document_id}')
    def delete(document_id: str, request: Request):
        request.app.state.rag.delete(document_id)
        return {'ok': True, 'data': {'deleted': document_id}}

    @app.post('/api/reindex')
    def reindex(request: Request):
        return {'ok': True, 'data': request.app.state.rag.reindex()}

    @app.post('/api/chat')
    def chat(body: ChatRequest, request: Request):
        return {'ok': True, 'data': request.app.state.rag.chat(body.question)}

    return app


app = create_app()
