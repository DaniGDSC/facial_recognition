"""REST API layer with authentication middleware for the facial recognition system.

Endpoints:
  POST   /api/auth/login          Operator login (returns session token)
  POST   /api/auth/logout         Revoke session
  GET    /api/auth/session        Verify current session

  POST   /api/recognize           Recognize a face from uploaded image
  POST   /api/authenticate        Authenticate via face (returns session token)
  POST   /api/enroll              Enroll a new user (requires operator role)

  GET    /api/health              Health check
  GET    /api/stats               System statistics (requires operator role)
  POST   /api/migrations/run      Run pending DB migrations (requires operator role)
"""

import base64
import logging
import os
import sys
import time
from functools import wraps
from http import HTTPStatus

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from flask import Flask, jsonify, request, g

from rbac import RBACManager
from session import SessionManager
from rate_limiter import RateLimiter
from recognition_constants import HIGH_SECURITY_THRESHOLD

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

    # Shared services
    app.extensions['session_mgr'] = SessionManager()
    app.extensions['rate_limiter'] = RateLimiter()
    app.extensions['rbac'] = RBACManager()
    app.extensions['recognition_system'] = None  # Lazy init

    # ---------- Middleware ----------

    @app.before_request
    def extract_auth():
        """Extract and verify bearer token on every request."""
        g.session = None
        g.user = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            session_mgr = app.extensions['session_mgr']
            session_data = session_mgr.verify_token(token)
            if session_data:
                g.session = session_data
                g.user = session_data

    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cache-Control'] = 'no-store'
        return response

    # ---------- Auth decorators ----------

    def require_session(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if g.session is None:
                return jsonify({'error': 'Authentication required'}), HTTPStatus.UNAUTHORIZED
            return f(*args, **kwargs)
        return wrapper

    def require_operator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if g.session is None:
                return jsonify({'error': 'Authentication required'}), HTTPStatus.UNAUTHORIZED
            # Re-verify operator role from RBAC DB (handles role changes/deletions)
            rbac = app.extensions['rbac']
            op_id = g.session.get('user_code')
            current_role = rbac.get_operator_role(op_id) if op_id else None
            if current_role not in ('OPERATOR', 'ADMIN'):
                return jsonify({'error': 'Operator access required'}), HTTPStatus.FORBIDDEN
            return f(*args, **kwargs)
        return wrapper

    def check_rate_limit():
        limiter = app.extensions['rate_limiter']
        client_ip = request.remote_addr or 'unknown'
        if not limiter.is_allowed(client_ip):
            remaining = limiter.cooldown_remaining(client_ip)
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': remaining,
            }), HTTPStatus.TOO_MANY_REQUESTS
        return None

    def get_recognition_system():
        if app.extensions['recognition_system'] is None:
            from recognition_controller import FacialRecognitionSystem
            app.extensions['recognition_system'] = FacialRecognitionSystem(
                similarity_threshold=0.8, use_faiss=True
            )
        return app.extensions['recognition_system']

    # ---------- Routes ----------

    @app.route('/api/health', methods=['GET'])
    def health():
        rec_sys = app.extensions.get('recognition_system')
        return jsonify({
            'status': 'healthy',
            'active_sessions': app.extensions['session_mgr'].active_session_count,
            'faiss_loaded': rec_sys.faiss_index is not None if rec_sys else False,
            'timestamp': int(time.time()),
        })

    @app.route('/api/auth/login', methods=['POST'])
    def operator_login():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'JSON body required'}), HTTPStatus.BAD_REQUEST

        operator_id = data.get('operator_id', '').strip()
        pin = data.get('pin', '')
        if not operator_id or not pin:
            return jsonify({'error': 'operator_id and pin required'}), HTTPStatus.BAD_REQUEST

        rbac = app.extensions['rbac']
        if not rbac.authenticate_operator(operator_id, pin):
            return jsonify({'error': 'Invalid credentials'}), HTTPStatus.UNAUTHORIZED

        session_mgr = app.extensions['session_mgr']
        token = session_mgr.create_session(
            user_id=operator_id, user_code=operator_id,
            full_name=operator_id, security_level='OPERATOR',
        )
        return jsonify({
            'token': token, 'operator_id': operator_id,
            'role': rbac.current_role, 'expires_in': session_mgr.session_ttl,
        })

    @app.route('/api/auth/logout', methods=['POST'])
    @require_session
    def logout():
        token = request.headers.get('Authorization', '')[7:]
        app.extensions['session_mgr'].revoke_token(token)
        return jsonify({'message': 'Logged out'})

    @app.route('/api/auth/session', methods=['GET'])
    @require_session
    def get_session():
        return jsonify(g.session)

    @app.route('/api/recognize', methods=['POST'])
    def recognize():
        rate_error = check_rate_limit()
        if rate_error:
            return rate_error

        image_data = _extract_image(request)
        if image_data is None:
            return jsonify({'error': 'Image required (file upload or base64 JSON)'}), HTTPStatus.BAD_REQUEST

        limiter = app.extensions['rate_limiter']
        limiter.record_attempt(request.remote_addr or 'unknown')

        rec_sys = get_recognition_system()
        client_ip = request.remote_addr or 'unknown'
        result = rec_sys.recognize_face(image_data, enable_liveness=True, top_k=5, client_ip=client_ip)

        # Redact user identity when no session — prevent reconnaissance
        recognized = result.get('recognized', False)
        user_info = result.get('user_info')
        if g.session and recognized and user_info:
            user_data = {'name': user_info['full_name'], 'user_code': user_info['user_code']}
        else:
            user_data = None

        return jsonify({
            'success': result.get('success', False),
            'recognized': recognized,
            'user': user_data,
            'liveness': {
                'passed': result.get('liveness_result', {}).get('is_live', False),
                'confidence': result.get('liveness_result', {}).get('confidence', 0.0),
            },
            'processing_time': round(result.get('processing_time', 0), 3),
        })

    @app.route('/api/authenticate', methods=['POST'])
    def authenticate():
        rate_error = check_rate_limit()
        if rate_error:
            return rate_error

        image_data = _extract_image(request)
        if image_data is None:
            return jsonify({'error': 'Image required'}), HTTPStatus.BAD_REQUEST

        security_level = request.args.get('security', 'STANDARD').upper()
        threshold = HIGH_SECURITY_THRESHOLD if security_level == 'HIGH' else 0.8

        limiter = app.extensions['rate_limiter']
        limiter.record_attempt(request.remote_addr or 'unknown')

        rec_sys = get_recognition_system()
        original_threshold = rec_sys.similarity_threshold
        try:
            rec_sys.similarity_threshold = threshold
            client_ip = request.remote_addr or 'unknown'
            result = rec_sys.recognize_face(image_data, enable_liveness=True, top_k=1, client_ip=client_ip)
        finally:
            rec_sys.similarity_threshold = original_threshold

        if not result.get('success') or not result.get('recognized'):
            return jsonify({'authenticated': False, 'message': 'Access denied'}), HTTPStatus.UNAUTHORIZED

        user_info = result['user_info']
        session_mgr = app.extensions['session_mgr']
        token = session_mgr.create_session(
            user_id=user_info['user_id'], user_code=user_info['user_code'],
            full_name=user_info['full_name'], security_level=security_level,
        )
        return jsonify({
            'authenticated': True, 'token': token,
            'user': {'name': user_info['full_name'], 'user_code': user_info['user_code']},
            'security_level': security_level, 'expires_in': session_mgr.session_ttl,
        })

    @app.route('/api/enroll', methods=['POST'])
    @require_operator
    def enroll():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'JSON body required'}), HTTPStatus.BAD_REQUEST

        user_code = data.get('user_code', '').strip()
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        image_b64 = data.get('image')

        if not all([user_code, first_name, last_name, image_b64]):
            return jsonify({
                'error': 'Required: user_code, first_name, last_name, image (base64)'
            }), HTTPStatus.BAD_REQUEST

        image_data = _decode_base64_image(image_b64)
        if image_data is None:
            return jsonify({'error': 'Invalid base64 image'}), HTTPStatus.BAD_REQUEST

        from enrollment_controller import UserEnrollmentSystem
        enrollment = UserEnrollmentSystem()
        try:
            client_ip = request.remote_addr or 'unknown'
            result = enrollment.enroll_with_image(user_code, first_name, last_name, image_data, client_ip)
        finally:
            enrollment.close()

        status = HTTPStatus.CREATED if result['success'] else HTTPStatus.CONFLICT
        return jsonify(result), status

    @app.route('/api/stats', methods=['GET'])
    @require_operator
    def stats():
        rec_sys = get_recognition_system()
        try:
            cursor = rec_sys.metadata_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE account_status = 'ACTIVE'")
            total_users = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM biometric_audit_log WHERE timestamp > ?",
                          (int(time.time()) - 86400,))
            recent = cursor.fetchone()[0]
            return jsonify({
                'enrolled_users': total_users, 'events_24h': recent,
                'faiss_active': rec_sys.faiss_index is not None,
                'active_sessions': app.extensions['session_mgr'].active_session_count,
            })
        except Exception:
            logger.exception("Stats query failed")
            return jsonify({'error': 'Failed to load statistics'}), HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route('/api/migrations/run', methods=['POST'])
    @require_operator
    def run_migrations():
        try:
            from migrations import SQLiteMigrator, PostgreSQLMigrator
            sm = SQLiteMigrator()
            sc = sm.migrate()
            sm.close()
            pm = PostgreSQLMigrator()
            pc = pm.migrate()
            pm.close()
            return jsonify({'sqlite_migrations': sc, 'pg_migrations': pc, 'message': 'Migrations complete'})
        except Exception as e:
            logger.exception("Migration failed")
            return jsonify({'error': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    return app


def _extract_image(req) -> np.ndarray | None:
    import cv2
    if 'image' in req.files:
        file_bytes = np.frombuffer(req.files['image'].read(), np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    data = req.get_json(silent=True)
    if data and 'image' in data:
        return _decode_base64_image(data['image'])
    return None


def _decode_base64_image(b64_string: str) -> np.ndarray | None:
    import cv2
    try:
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=False)
