import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import prometheus_client as prom

# Metrics
AUTH_LATENCY = prom.Histogram('auth_latency_seconds', 'Authentication latency')
AUTH_SUCCESS = prom.Counter('auth_success_total', 'Successful authentications')
AUTH_FAILURES = prom.Counter('auth_failures_total', 'Failed authentications', ['reason'])

@dataclass
class AuthRequest:
    """Structured auth request"""
    request_id: str
    timestamp: datetime
    security_level: str
    enable_liveness: bool
    timeout: float = 30.0

class ProductionRecognitionInterface:
    """Production-ready async interface"""
    
    def __init__(self, recognition_system, max_concurrent=10):
        self.system = recognition_system
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)
        
    @AUTH_LATENCY.time()
    async def authenticate_async(self, request: AuthRequest) -> dict:
        """Async authentication with timeout and monitoring"""
        async with self.semaphore:  # Limit concurrency
            try:
                # Timeout protection
                result = await asyncio.wait_for(
                    self._do_authentication(request),
                    timeout=request.timeout
                )
                
                if result.get('authenticated'):
                    AUTH_SUCCESS.inc()
                else:
                    AUTH_FAILURES.labels(reason='not_recognized').inc()
                
                return result
                
            except asyncio.TimeoutError:
                AUTH_FAILURES.labels(reason='timeout').inc()
                self.logger.error(f"Auth timeout: {request.request_id}")
                return {'success': False, 'error': 'timeout'}
                
            except Exception as e:
                AUTH_FAILURES.labels(reason='error').inc()
                self.logger.exception(f"Auth error: {request.request_id}")
                return {'success': False, 'error': str(e)}
    
    async def _do_authentication(self, request: AuthRequest) -> dict:
        """Actual auth logic (run in thread pool for CPU-bound work)"""
        loop = asyncio.get_event_loop()
        
        # Run blocking camera/recognition in thread pool
        result = await loop.run_in_executor(
            None,  # Use default executor
            self.system.recognize_from_camera,
            request.enable_liveness,
            1  # top_k
        )
        
        return build_auth_result(result, request.security_level)
    
    async def health_check(self) -> dict:
        """Health check endpoint"""
        try:
            # Check database
            db_ok = await self._check_database()
            
            # Check FAISS index
            index_ok = self.system.faiss_index is not None
            
            # Check camera
            camera_ok = await self._check_camera()
            
            healthy = db_ok and index_ok and camera_ok
            
            return {
                'status': 'healthy' if healthy else 'degraded',
                'database': db_ok,
                'faiss_index': index_ok,
                'camera': camera_ok,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.exception("Health check failed")
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _check_database(self) -> bool:
        """Check database connection (dummy implementation)"""
        await asyncio.sleep(0.1)  # Simulate async DB check
        return True  # Assume DB is always ok for this example
    
    async def _check_camera(self) -> bool:
        """Check camera status (dummy implementation)"""
        await asyncio.sleep(0.1)  # Simulate async camera check
        return True  # Assume camera is always ok for this example

def build_auth_result(recognition_result, security_level) -> dict:
    """Build structured auth result from recognition system output"""
    # Dummy implementation: always return success with dummy user data
    return {
        'authenticated': True,
        'user_id': '12345',
        'security_level': security_level,
        'attributes': {
            'name': 'John Doe',
            'role': 'admin'
        }
    }