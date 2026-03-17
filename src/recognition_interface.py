import getpass
import logging
import sys

from recognition_controller import FacialRecognitionSystem
from recognition_constants import (
    UNIFORM_ACCESS_DENIED,
    UNIFORM_OPERATION_FAIL,
    MAX_INPUT_ATTEMPTS,
    HIGH_SECURITY_THRESHOLD,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
)
from rate_limiter import RateLimiter
from rbac import RBACManager, ROLE_ADMIN

logger = logging.getLogger(__name__)

# Module-level rate limiter instance
_rate_limiter = RateLimiter()
_rbac: RBACManager | None = None


def setup_logging():
    """Configure structured logging for the application."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("facial_recognition.log", encoding="utf-8"),
        ],
    )


def display_recognition_menu():
    """Display camera recognition menu"""
    role = _rbac.current_role if _rbac else None
    operator = _rbac.current_operator if _rbac else None
    print("\n" + "=" * 50)
    print("    FACIAL RECOGNITION SYSTEM")
    if operator:
        print(f"    Operator: {operator} ({role})")
    print("=" * 50)
    print("1.  Identify Person (1:N Recognition)")
    print("2.  Authenticate Access (Pure Face Auth)")
    print("3.  Secure Access (High Security)")
    print("4.  View Statistics")
    print("5.  Operator Login / Logout")
    print("6.  Exit")
    print("=" * 50)


def get_validated_input(prompt, validator, error_msg, max_attempts=MAX_INPUT_ATTEMPTS, allow_empty=False, default=None):
    """Generic input with validation and attempt limit."""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = input(prompt).strip()
            if raw == "":
                if allow_empty:
                    return default
                print("Input required")
                continue
            if validator(raw):
                return raw
            remaining = max_attempts - attempt
            if remaining > 0:
                print(f"{error_msg} ({remaining} attempt{'s' if remaining > 1 else ''} remaining)")
            else:
                print(f"{error_msg}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("Input error: %s", e)
    print(f"Maximum attempts ({max_attempts}) exceeded")
    return None


def get_integer_input(prompt, min_val, max_val, default=None, max_attempts=MAX_INPUT_ATTEMPTS):
    """Integer input within a range with attempt limit."""
    default_str = f" [{default}]" if default is not None else ""
    full_prompt = f"{prompt} ({min_val}-{max_val}){default_str}: "

    def validator(inp):
        try:
            v = int(inp)
            return min_val <= v <= max_val
        except ValueError:
            return False

    result = get_validated_input(
        full_prompt, validator,
        f"Enter a number between {min_val} and {max_val}",
        max_attempts, allow_empty=(default is not None),
        default=str(default) if default is not None else None
    )

    if result is None:
        return None
    return int(result) if result != "" else default


def get_menu_choice(min_choice, max_choice, max_attempts=MAX_INPUT_ATTEMPTS):
    """Menu choice as string; None if invalid after attempts."""
    def validator(inp):
        return inp.isdigit() and min_choice <= int(inp) <= max_choice

    return get_validated_input(
        f"Select option ({min_choice}-{max_choice}): ",
        validator,
        f"Please enter a number between {min_choice} and {max_choice}",
        max_attempts, allow_empty=False
    )


def format_liveness_status(result, liveness_enabled=None):
    liveness_result = result.get('liveness_result')
    if not liveness_result:
        return "N/A (ERROR)"
    is_live = liveness_result.get('is_live', False)
    confidence = liveness_result.get('confidence')
    status = "PASS" if is_live else "FAIL"
    return f"{status} (conf: {confidence:.2f})" if isinstance(confidence, (int, float)) else status


def _check_rate_limit(identifier: str = "default") -> bool:
    """Check rate limit before an authentication attempt. Returns True if allowed."""
    if not _rate_limiter.is_allowed(identifier):
        remaining = _rate_limiter.cooldown_remaining(identifier)
        print(f"\nRate limit exceeded. Try again in {remaining}s.")
        logger.warning("Authentication rate-limited for identifier=%s", identifier)
        return False
    return True


def recognize_from_camera(recognition_system):
    """Handle camera-based face identification"""
    if not _check_rate_limit():
        return

    print("\nFACE IDENTIFICATION MODE")
    print("=" * 40)

    top_k = get_integer_input("Top matches to show", 1, 10, default=5)
    if top_k is None:
        print("Operation cancelled - invalid input")
        return

    enable_liveness = True
    logger.info("Starting face identification (liveness=enforced, top_k=%d)", top_k)
    print(f"\nStarting face identification...")
    print(f"  Liveness: ENFORCED")
    print(f"  Top matches: {top_k}")
    print("  Press SPACE to capture, ESC to cancel\n")

    try:
        _rate_limiter.record_attempt("default")
        result = recognition_system.recognize_from_camera(enable_liveness, top_k)
        result['_liveness_enabled'] = enable_liveness
        display_recognition_results(result)
    except KeyboardInterrupt:
        print("\nRecognition cancelled")
    except Exception:
        logger.exception("Recognition from camera failed")
        print(f"\n{UNIFORM_OPERATION_FAIL}")


def build_auth_result(result, security_level='STANDARD', required_threshold=None):
    """Standardize recognition result for auth-related displays."""
    success = result.get('success', False)
    recognized = result.get('recognized', False)
    auth_result = {
        'success': success,
        'authenticated': bool(success and recognized),
        'user_info': result.get('user_info') if recognized else None,
        'similarity_score': result.get('similarity_score', 0.0),
        'liveness_result': result.get('liveness_result'),
        'processing_time': result.get('processing_time', 0.0),
        'search_method': result.get('search_method', 'Unknown'),
        'message': UNIFORM_ACCESS_DENIED if success and not recognized else (UNIFORM_OPERATION_FAIL if not success else "Welcome")
    }
    if security_level == 'HIGH':
        auth_result['security_level'] = 'HIGH'
        auth_result['required_threshold'] = required_threshold
    return auth_result


def authenticate_user_pure(recognition_system):
    """Authentication without user code."""
    if not _check_rate_limit():
        return

    print("\nFACIAL AUTHENTICATION")
    print("=" * 40)
    enable_liveness = True
    remaining = _rate_limiter.remaining_attempts("default")
    logger.info("Starting facial authentication (attempts_remaining=%d)", remaining)
    print(f"\nAuthenticating via facial recognition...")
    print(f"   Liveness: ON")
    print(f"   Attempts remaining: {remaining}")
    print("Position face in camera and press SPACE")

    try:
        _rate_limiter.record_attempt("default")
        result = recognition_system.recognize_from_camera(enable_liveness, top_k=1)
        auth_result = build_auth_result(result, security_level='STANDARD')
        auth_result['_liveness_enabled'] = enable_liveness
        display_authentication_results(auth_result)
    except KeyboardInterrupt:
        print("\nAuthentication cancelled")
    except Exception:
        logger.exception("Pure authentication failed")
        print(f"\n{UNIFORM_OPERATION_FAIL}")


def authenticate_secure_access(recognition_system):
    """High-security authentication with elevated threshold."""
    if not _check_rate_limit():
        return

    print("\nSECURE ACCESS AUTHENTICATION")
    print("=" * 40)
    print("High security mode - Higher confidence required")
    threshold = HIGH_SECURITY_THRESHOLD
    remaining = _rate_limiter.remaining_attempts("default")
    logger.info("Starting secure authentication (threshold=%.2f, attempts_remaining=%d)",
                threshold, remaining)
    print(f"\nSecure authentication starting...")
    print(f"   Confidence required: {threshold:.1%}")
    print(f"   Liveness: MANDATORY")
    print(f"   Attempts remaining: {remaining}")
    print("Position face in camera and press SPACE")

    original_threshold = getattr(recognition_system, 'similarity_threshold', threshold)
    try:
        recognition_system.similarity_threshold = threshold
        _rate_limiter.record_attempt("default")
        result = recognition_system.recognize_from_camera(enable_liveness=True, top_k=1)
        auth_result = build_auth_result(result, security_level='HIGH', required_threshold=threshold)
        auth_result['_liveness_enabled'] = True
        display_secure_authentication_results(auth_result)
    except KeyboardInterrupt:
        print("\nSecure authentication cancelled")
    except Exception:
        logger.exception("Secure authentication failed")
        print(f"\n{UNIFORM_OPERATION_FAIL}")
    finally:
        try:
            recognition_system.similarity_threshold = original_threshold
        except Exception as e:
            logger.error("Failed to restore similarity threshold: %s", e)


def display_recognition_results(result):
    print("\n" + "=" * 50)
    print("         IDENTIFICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"{UNIFORM_OPERATION_FAIL}")
        return
    liveness_status = format_liveness_status(result, result.get('_liveness_enabled'))
    print(f"  Liveness: {liveness_status}")
    if result.get('recognized'):
        user = result.get('user_info') or {}
        print(f"\nPERSON IDENTIFIED!")
        print(f"   Name: {user.get('full_name', 'Unknown')}")
    else:
        print(f"\nUNKNOWN PERSON")
        print(f"   {UNIFORM_ACCESS_DENIED}")
    print("=" * 50)


def display_authentication_results(result):
    """Display standard authentication results."""
    print("\n" + "=" * 50)
    print("       AUTHENTICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"{UNIFORM_OPERATION_FAIL}")
        return
    print(f"  Liveness: {format_liveness_status(result, result.get('_liveness_enabled'))}")
    if result.get('authenticated'):
        user = result.get('user_info') or {}
        print(f"\nAUTHENTICATION SUCCESS!")
        print(f"   Welcome: {user.get('full_name', 'User')}")
        print(f"   Access: GRANTED")
    else:
        print(f"\n{UNIFORM_ACCESS_DENIED}")
        print(f"   Access: DENIED")
    print("=" * 50)


def display_secure_authentication_results(result):
    print("\n" + "=" * 50)
    print("      SECURE AUTHENTICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"{UNIFORM_OPERATION_FAIL}")
        return
    print(f"  Security: {result.get('security_level', 'STANDARD')}")
    print(f"  Liveness: {format_liveness_status(result, True)}")
    if result.get('authenticated'):
        user = result.get('user_info') or {}
        print(f"\nSECURE ACCESS GRANTED!")
        print(f"   Authorized: {user.get('full_name', 'User')}")
        print(f"   Access: HIGH SECURITY GRANTED")
    else:
        print(f"\n{UNIFORM_ACCESS_DENIED}")
        print(f"   Access: HIGH SECURITY DENIED")
    print("=" * 50)


def operator_login_logout():
    """Handle operator login/logout and initial setup."""
    global _rbac
    if _rbac is None:
        _rbac = RBACManager()

    if _rbac.current_operator:
        print(f"\nCurrently logged in as: {_rbac.current_operator} ({_rbac.current_role})")
        choice = input("Logout? (Y/n) [Y]: ").strip().lower()
        if choice in ['', 'y', 'yes']:
            _rbac.logout()
            print("Logged out successfully")
        return

    # First-run setup
    if not _rbac.has_operators():
        print("\nNo operator accounts found. Creating initial admin account.")
        op_id = input("Enter operator ID: ").strip()
        if not op_id:
            print("Cancelled")
            return
        pin = getpass.getpass("Enter PIN: ")
        pin_confirm = getpass.getpass("Confirm PIN: ")
        if pin != pin_confirm:
            print("PINs do not match")
            return
        if len(pin) < 4:
            print("PIN must be at least 4 characters")
            return
        if _rbac.register_operator(op_id, pin, role=ROLE_ADMIN):
            print(f"Admin account '{op_id}' created. Please log in.")
        else:
            print("Failed to create account")
        return

    # Normal login
    print("\nOPERATOR LOGIN")
    print("=" * 30)
    op_id = input("Operator ID: ").strip()
    if not op_id:
        print("Cancelled")
        return
    pin = getpass.getpass("PIN: ")
    if _rbac.authenticate_operator(op_id, pin):
        print(f"Welcome, {op_id} ({_rbac.current_role})")
    else:
        print("Authentication failed")


def view_statistics(recognition_system):
    print("\nSYSTEM STATISTICS")
    print("=" * 40)
    try:
        cursor = recognition_system.metadata_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE account_status = 'ACTIVE'")
        total_users = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM biometric_audit_log
            WHERE timestamp > (CAST(strftime('%s','now') AS INTEGER) - ?)
        """, (SECONDS_PER_DAY,))
        recent_activity = cursor.fetchone()[0]

        cursor.execute("""
            SELECT
                COUNT(CASE WHEN event_type LIKE '%Success' THEN 1 END) as successes,
                COUNT(*) as total
            FROM biometric_audit_log
            WHERE timestamp > (CAST(strftime('%s','now') AS INTEGER) - ?)
        """, (SECONDS_PER_WEEK,))
        successes, total = cursor.fetchone()
        success_rate = (successes / total * 100) if total else 0.0

        print(f"  Enrolled Users: {total_users}")
        print(f"  24h Activity: {recent_activity} events")
        print(f"  7-Day Success Rate: {success_rate:.1f}%")
        print(f"  Search Method: {'FAISS' if getattr(recognition_system, 'use_faiss', False) else 'Linear'}")
        print(f"  FAISS Index: {'Active' if getattr(recognition_system, 'faiss_index', None) else 'Not built'}")
        print(f"  Similarity Threshold: {getattr(recognition_system, 'similarity_threshold', 0.8):.3f}")
        print(f"  Rate Limit: {_rate_limiter.remaining_attempts('default')} attempts remaining")
    except Exception:
        logger.exception("Failed to load statistics")
        print(f"{UNIFORM_OPERATION_FAIL}")


def main():
    """Main loop."""
    setup_logging()

    recognition_system = None
    try:
        logger.info("Initializing Facial Recognition System...")
        recognition_system = FacialRecognitionSystem(similarity_threshold=0.8, use_faiss=True)
        logger.info("System ready")

        _rbac = RBACManager()

        while True:
            display_recognition_menu()
            choice = get_menu_choice(1, 6)
            if choice is None:
                print("Too many invalid inputs - returning to menu")
                continue
            if choice == '1':
                recognize_from_camera(recognition_system)
            elif choice == '2':
                authenticate_user_pure(recognition_system)
            elif choice == '3':
                authenticate_secure_access(recognition_system)
            elif choice == '4':
                if _rbac.check_permission("view_statistics"):
                    view_statistics(recognition_system)
                else:
                    print("\nOperator login required to view statistics")
            elif choice == '5':
                operator_login_logout()
            elif choice == '6':
                print("Goodbye!")
                break
            input("\nPress ENTER to continue...")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception:
        logger.exception("Fatal error in main loop")
        print(f"{UNIFORM_OPERATION_FAIL}")
    finally:
        try:
            if recognition_system:
                recognition_system.close()
        except Exception as e:
            logger.error("Failed to close recognition system: %s", e)


if __name__ == "__main__":
    main()
