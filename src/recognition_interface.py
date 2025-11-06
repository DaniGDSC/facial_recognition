from recognition_controller import FacialRecognitionSystem
from recognition_constants import (
    UNIFORM_ACCESS_DENIED, 
    UNIFORM_OPERATION_FAIL, 
    MAX_INPUT_ATTEMPTS,
    HIGH_SECURITY_THRESHOLD,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    MAX_AUTH_ATTEMPTS_PER_HOUR,
    COOLDOWN_PERIOD_SECONDS
)

def display_recognition_menu():
    """Display camera recognition menu"""
    print("\n" + "=" * 50)
    print("    FACIAL RECOGNITION SYSTEM")
    print("=" * 50)
    print("1. 👁️  Identify Person (1:N Recognition)")
    print("2. 🔐 Authenticate Access (Pure Face Auth)")
    print("3. 🔒 Secure Access (High Security)")
    print("4. 📊 View Statistics")
    print("5. 🚪 Exit")
    print("=" * 50)

def get_validated_input(prompt, validator, error_msg, max_attempts=MAX_INPUT_ATTEMPTS, allow_empty=False, default=None):
    """Generic input with validation and attempt limit."""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = input(prompt).strip()
            if raw == "":
                if allow_empty:
                    return default
                print("❌ Input required")
                continue
            if validator(raw):
                return raw
            remaining = max_attempts - attempt
            if remaining > 0:
                print(f"❌ {error_msg} ({remaining} attempt{'s' if remaining > 1 else ''} remaining)")
            else:
                print(f"❌ {error_msg}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"❌ Input error: {e}")
    print(f"⚠️  Maximum attempts ({max_attempts}) exceeded")
    return None


def get_yes_no_input(prompt, default=True, max_attempts=MAX_INPUT_ATTEMPTS):
    """Yes/No input with validation and default."""
    default_str = "Y" if default else "n"
    full_prompt = f"{prompt} (Y/n) [{default_str}]: "
    
    def validator(inp):
        return inp.lower() in ["y", "yes", "n", "no"]
    
    result = get_validated_input(
        full_prompt,
        validator,
        "Please enter 'Y' or 'N'",
        max_attempts,
        allow_empty=True,
        default="y" if default else "n"
    )
    
    if result is None:
        return None
    
    return result.lower() in ["y", "yes"]


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
        full_prompt,
        validator,
        f"Enter a number between {min_val} and {max_val}",
        max_attempts,
        allow_empty=(default is not None),
        default=str(default) if default is not None else None
    )
    
    if result is None:
        return None
    
    return int(result) if result != "" else default


def get_menu_choice(min_choice, max_choice, max_attempts=MAX_INPUT_ATTEMPTS):
    """Menu choice as string; None if invalid after attempts."""
    def validator(inp):
        return inp.isdigit() and min_choice <= int(inp) <= max_choice
    
    result = get_validated_input(
        f"Select option ({min_choice}-{max_choice}): ",
        validator,
        f"Please enter a number between {min_choice} and {max_choice}",
        max_attempts,
        allow_empty=False
    )
    
    return result


def format_liveness_status(result, liveness_enabled=None):
    """Return liveness status string with Disabled/N/A/Pass/Fail."""
    liveness_result = result.get('liveness_result')
    if not liveness_result:
        return "N/A (ERROR)"
    
    is_live = liveness_result.get('is_live', False)
    confidence = liveness_result.get('confidence')
    status = "✓ PASS" if is_live else "✗ FAIL"
    return f"{status} (conf: {confidence:.2f})" if isinstance(confidence, (int, float)) else status


def recognize_from_camera(recognition_system):
    """Handle camera-based face identification"""
    print("\n📷 FACE IDENTIFICATION MODE")
    print("=" * 40)
    
    # Get liveness preference with validation
    enable_liveness = True
    if enable_liveness is None:
        print("⚠️  Operation cancelled - invalid input")
        return

    # Get number of matches to show with validation
    top_k = get_integer_input("Top matches to show", 1, 10, default=5)
    if top_k is None:
        print("⚠️  Operation cancelled - invalid input")
        return
    
    print(f"\n🎯 Starting face identification...")
    print(f"  Liveness: ✓ ENFORCED")
    print(f"  Top matches: {top_k}")
    print("   Press SPACE to capture, ESC to cancel\n")
    
    try:
        result = recognition_system.recognize_from_camera(enable_liveness, top_k)
        result['_liveness_enabled'] = enable_liveness
        display_recognition_results(result)
    except KeyboardInterrupt:
        print("\n⚠️  Recognition cancelled")
    except Exception:
        print(f"\n❌ {UNIFORM_OPERATION_FAIL}")


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


def get_liveness_preference(mandatory=False):
    return True

def authenticate_user_pure(recognition_system):
    """Authentication without user code."""
    print("\n🔐 FACIAL AUTHENTICATION")
    print("=" * 40)
    enable_liveness = get_liveness_preference(mandatory=False)
    if enable_liveness is None:
        print("⚠️  Operation cancelled - invalid input")
        return
    print(f"\n🎯 Authenticating via facial recognition...")
    print(f"   Liveness: {'✓ ON' if enable_liveness else '✗ OFF'}")
    print("📷 Position face in camera and press SPACE")
    try:
        result = recognition_system.recognize_from_camera(enable_liveness, top_k=1)
        auth_result = build_auth_result(result, security_level='STANDARD')
        auth_result['_liveness_enabled'] = enable_liveness
        display_authentication_results(auth_result)
    except KeyboardInterrupt:
        print("\n⚠️  Authentication cancelled")
    except Exception:
        print(f"\n❌ {UNIFORM_OPERATION_FAIL}")


def authenticate_secure_access(recognition_system):
    """High-security authentication with elevated threshold."""
    print("\n🔒 SECURE ACCESS AUTHENTICATION")
    print("=" * 40)
    print("🛡️  High security mode - Higher confidence required")
    enable_liveness = get_liveness_preference(mandatory=True)
    HIGH_SECURITY_THRESHOLD = 0.85
    print(f"\n🎯 Secure authentication starting...")
    print(f"   Confidence required: {HIGH_SECURITY_THRESHOLD:.1%}")
    print(f"   Liveness: ✓ MANDATORY")
    print("📷 Position face in camera and press SPACE")
    original_threshold = getattr(recognition_system, 'similarity_threshold', HIGH_SECURITY_THRESHOLD)
    try:
        recognition_system.similarity_threshold = HIGH_SECURITY_THRESHOLD
        result = recognition_system.recognize_from_camera(enable_liveness=True, top_k=1)
        auth_result = build_auth_result(result, security_level='HIGH', required_threshold=HIGH_SECURITY_THRESHOLD)
        auth_result['_liveness_enabled'] = True
        display_secure_authentication_results(auth_result)
    except KeyboardInterrupt:
        print("\n⚠️  Secure authentication cancelled")
    except Exception:
        print(f"\n❌ {UNIFORM_OPERATION_FAIL}")
    finally:
        try:
            recognition_system.similarity_threshold = original_threshold
        except Exception:
            pass

def display_recognition_results(result):
    print("\n" + "=" * 50)
    print("         IDENTIFICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    # Removed raw similarity/match leakage
    liveness_status = format_liveness_status(result, result.get('_liveness_enabled'))
    print(f"❤️  Liveness: {liveness_status}")
    if result.get('recognized'):
        user = result.get('user_info') or {}
        print(f"\n🎉 PERSON IDENTIFIED!")
        print(f"   👤 Name: {user.get('full_name', 'Unknown')}")
        # Optionally could show similarity internally; omitted for security
    else:
        print(f"\n❓ UNKNOWN PERSON")
        print(f"   💭 {UNIFORM_ACCESS_DENIED}")
    # Top matches removed to prevent threshold probing
    print("=" * 50)

def display_authentication_results(result):
    """Display standard authentication results (hide score unless authenticated)."""
    print("\n" + "=" * 50)
    print("       AUTHENTICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    print(f"❤️  Liveness: {format_liveness_status(result, result.get('_liveness_enabled'))}")
    if result.get('authenticated'):
        user = result.get('user_info') or {}
        print(f"\n🎉 AUTHENTICATION SUCCESS!")
        print(f"   👤 Welcome: {user.get('full_name', 'User')}")
        print(f"   🚪 Access: ✅ GRANTED")
        # Score intentionally not shown to user
    else:
        print(f"\n❌ {UNIFORM_ACCESS_DENIED}")
        print(f"   🚪 Access: ⛔ DENIED")
    print("=" * 50)

def display_secure_authentication_results(result):
    print("\n" + "=" * 50)
    print("      SECURE AUTHENTICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    print(f"️  Security: {result.get('security_level', 'STANDARD')}")
    print(f"❤️  Liveness: {format_liveness_status(result, True)}")
    if result.get('authenticated'):
        user = result.get('user_info') or {}
        print(f"\n🔒 SECURE ACCESS GRANTED!")
        print(f"   👤 Authorized: {user.get('full_name', 'User')}")
        print(f"   🚪 Access: ✅ HIGH SECURITY GRANTED")
    else:
        print(f"\n🚫 {UNIFORM_ACCESS_DENIED}")
        print(f"   🚪 Access: ⛔ HIGH SECURITY DENIED")
    print("=" * 50)


def view_statistics(recognition_system):
    print("\n📊 SYSTEM STATISTICS")
    print("=" * 40)
    try:
        cursor = recognition_system.metadata_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE account_status = 'ACTIVE'")
        total_users = cursor.fetchone()[0]
        # Last 24h
        cursor.execute("""
            SELECT COUNT(*) FROM biometric_audit_log 
            WHERE timestamp > (CAST(strftime('%s','now') AS INTEGER) - ?)
        """, (SECONDS_PER_DAY,))
        recent_activity = cursor.fetchone()[0]
        # Last 7 days
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN event_type LIKE '%Success' THEN 1 END) as successes,
                COUNT(*) as total
            FROM biometric_audit_log 
            WHERE timestamp > (CAST(strftime('%s','now') AS INTEGER) - ?)
        """, (SECONDS_PER_WEEK,))
        successes, total = cursor.fetchone()
        success_rate = (successes / total * 100) if total else 0.0
        print(f"👥 Enrolled Users: {total_users}")
        print(f"📈 24h Activity: {recent_activity} events")
        print(f"📊 7-Day Success Rate: {success_rate:.1f}%")
        print(f"🔍 Search Method: {'FAISS' if getattr(recognition_system, 'use_faiss', False) else 'Linear'}")
        print(f"⚡ FAISS Index: {'✓ Active' if getattr(recognition_system, 'faiss_index', None) else '✗ Not built'}")
        print(f"🎯 Similarity Threshold: {getattr(recognition_system, 'similarity_threshold', 0.8):.3f}")
    except Exception:
        print(f"❌ {UNIFORM_OPERATION_FAIL}")


def main():
    """Main loop."""
    recognition_system = None
    try:
        print("🚀 Initializing Facial Recognition System...")
        recognition_system = FacialRecognitionSystem(similarity_threshold=0.8, use_faiss=True)
        print(f"✅ System ready")
        while True:
            display_recognition_menu()
            choice = get_menu_choice(1, 5)
            if choice is None:
                print("⚠️  Too many invalid inputs - returning to menu")
                continue
            if choice == '1':
                recognize_from_camera(recognition_system)
            elif choice == '2':
                authenticate_user_pure(recognition_system)
            elif choice == '3':
                authenticate_secure_access(recognition_system)
            elif choice == '4':
                view_statistics(recognition_system)
            elif choice == '5':
                print("👋 Goodbye!")
                break
            input("\nPress ENTER to continue...")
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted")
    except Exception:
        print(f"💥 {UNIFORM_OPERATION_FAIL}")
    finally:
        try:
            if recognition_system:
                recognition_system.close()
                print("🔧 System closed")
        except Exception:
            pass

if __name__ == "__main__":
    main()