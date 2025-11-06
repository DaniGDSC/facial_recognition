from recognition_controller import FacialRecognitionSystem
from recognition_constants import UNIFORM_ACCESS_DENIED, UNIFORM_OPERATION_FAIL

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

def format_liveness_status(result, liveness_enabled=None):
    """
    Format liveness detection status for display.
    
    Args:
        result: Recognition result dictionary
        liveness_enabled: Optional bool indicating if liveness was enabled.
                         If None, will infer from result.
    
    Returns:
        str: Formatted liveness status
    """
    liveness_result = result.get('liveness_result')
    
    # If no liveness result exists, it wasn't checked
    if not liveness_result or liveness_result == {'is_live': False}:
        # Check if this is default value or actual failure
        # If liveness was explicitly disabled, show N/A
        if liveness_enabled is False:
            return "N/A (Disabled)"
        # If we can't determine, show N/A
        elif liveness_result is None:
            return "N/A (Not checked)"
        # Otherwise it's a real failure
        else:
            is_live = liveness_result.get('is_live', False)
            confidence = liveness_result.get('confidence', 0.0)
            if is_live:
                return f"✓ PASS (conf: {confidence:.2f})"
            else:
                return f"✗ FAIL (conf: {confidence:.2f})"
    
    # We have a real liveness result
    is_live = liveness_result.get('is_live', False)
    confidence = liveness_result.get('confidence')
    
    if is_live:
        if confidence is not None:
            return f"✓ PASS (conf: {confidence:.2f})"
        return "✓ PASS"
    else:
        if confidence is not None:
            return f"✗ FAIL (conf: {confidence:.2f})"
        return "✗ FAIL"


def recognize_from_camera(recognition_system):
    """Handle camera-based face identification"""
    print("\n📷 FACE IDENTIFICATION MODE")
    print("=" * 40)
    
    # Get liveness preference
    enable_liveness = get_liveness_preference(mandatory=False)

    # Get number of matches to show
    while True:
        try:
            top_k = input("Top matches to show (1-10) [5]: ").strip()
            if top_k == '':
                top_k = 5
                break
            else:
                top_k = int(top_k)
                if 1 <= top_k <= 10:
                    break
                else:
                    print("❌ Enter number between 1-10")
        except ValueError:
            print("❌ Enter valid number")
    
    print(f"\n🎯 Starting face identification...")
    print(f"   Liveness: {'✓ ON' if enable_liveness else '✗ OFF'}")
    print(f"   Top matches: {top_k}")
    print("   Press SPACE to capture, ESC to cancel\n")
    
    try:
        result = recognition_system.recognize_from_camera(enable_liveness, top_k)
        # Store liveness preference in result for display
        result['_liveness_enabled'] = enable_liveness
        display_recognition_results(result)
        
    except KeyboardInterrupt:
        print("\n⚠️  Recognition cancelled")

def build_auth_result(result, security_level='STANDARD', required_threshold=None):
    """
    Build standardized authentication result dictionary.
    
    Args:
        result: Raw recognition result from camera
        security_level: 'STANDARD' or 'HIGH'
        required_threshold: Threshold used for high security mode
    
    Returns:
        Standardized auth_result dictionary
    """
    success = result.get('success', False)
    recognized = result.get('recognized', False)
    
    auth_result = {
        'success': success,
        'authenticated': success and recognized,
        'user_info': result.get('user_info') if recognized else None,
        'similarity_score': result.get('similarity_score', 0.0),
        'liveness_result': result.get('liveness_result', {'is_live': False}),
        'processing_time': result.get('processing_time', 0.0),
        'search_method': result.get('search_method', 'Unknown'),
    }
    
    # Add security-specific fields
    if security_level == 'HIGH':
        auth_result['security_level'] = 'HIGH'
        auth_result['required_threshold'] = required_threshold
    
    # Set appropriate message
    if not success:
        auth_result['message'] = UNIFORM_OPERATION_FAIL
    elif recognized:
        user = result.get('user_info', {})
        if security_level == 'HIGH':
            auth_result['message'] = "Secure access granted"
        else:
            auth_result['message'] = f"Welcome {user.get('full_name', 'User')}"
    else:
        auth_result['message'] = UNIFORM_ACCESS_DENIED
    
    return auth_result


def get_liveness_preference(mandatory=False):
    if mandatory:
        print("🔐 Liveness detection: ✓ MANDATORY for secure access")
        return True
    
    while True:
        liveness_choice = input("Enable liveness detection? (Y/n) [Y]: ").strip().lower()
        if liveness_choice in ['', 'y', 'yes']:
            return True
        elif liveness_choice in ['n', 'no']:
            return False
        else:
            print("❌ Please enter 'Y' or 'N'")

def authenticate_user_pure(recognition_system):
    """Pure facial authentication - no user code required"""
    print("\n🔐 FACIAL AUTHENTICATION")
    print("=" * 40)
    print("📷 Look at camera to authenticate")
    
    # Get liveness preference using helper
    enable_liveness = get_liveness_preference(mandatory=False)
    
    print(f"\n🎯 Authenticating via facial recognition...")
    print(f"   Liveness: {'✓ ON' if enable_liveness else '✗ OFF'}")
    print("📷 Position face in camera and press SPACE")
    
    try:
        result = recognition_system.recognize_from_camera(enable_liveness, top_k=1)
        
        # Use helper function to build result
        auth_result = build_auth_result(result, security_level='STANDARD')
        # Store liveness preference for display
        auth_result['_liveness_enabled'] = enable_liveness
        
        display_authentication_results(auth_result)
    except KeyboardInterrupt:
        print("\n⚠️  Authentication cancelled")

def authenticate_secure_access(recognition_system):
    """High-security authentication with elevated threshold"""
    print("\n🔒 SECURE ACCESS AUTHENTICATION")
    print("=" * 40)
    print("🛡️  High security mode - Higher confidence required")
    
    # Get liveness preference (mandatory for secure access)
    enable_liveness = get_liveness_preference(mandatory=True)
    
    # Set high security threshold
    high_security_threshold = 0.85
    
    print(f"\n🎯 Secure authentication starting...")
    print(f"   Confidence required: {high_security_threshold:.1%}")
    print(f"   Liveness: ✓ MANDATORY")
    print("📷 Position face in camera and press SPACE")
    
    # Store original threshold to restore later
    original_threshold = getattr(recognition_system, 'similarity_threshold', high_security_threshold)
    
    try:
        # Temporarily set high security threshold
        recognition_system.similarity_threshold = high_security_threshold
        
        result = recognition_system.recognize_from_camera(enable_liveness=True, top_k=1)
        
        # Use helper function to build result
        auth_result = build_auth_result(
            result, 
            security_level='HIGH',
            required_threshold=high_security_threshold
        )
        # Store liveness preference for display (always True for secure)
        auth_result['_liveness_enabled'] = True
        
        display_secure_authentication_results(auth_result)
        
    except KeyboardInterrupt:
        print("\n⚠️  Secure authentication cancelled")
    except Exception as e:
        print(f"\n❌ {UNIFORM_OPERATION_FAIL}")
        # Optional: Log the actual error for debugging
        # print(f"Debug: {type(e).__name__}: {e}")
    finally:
        # Always restore original threshold
        try:
            recognition_system.similarity_threshold = original_threshold
        except AttributeError:
            # If similarity_threshold doesn't exist, ignore
            pass

def display_recognition_results(result):
    """Display identification results"""
    print("\n" + "=" * 50)
    print("         IDENTIFICATION RESULTS")
    print("=" * 50)
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    
    print(f"⏱️  Time: {result.get('processing_time', 0.0):.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"👥 Database: {result.get('database_size', 0)} users")
    
    # Use improved liveness formatting
    liveness_enabled = result.get('_liveness_enabled')
    liveness_status = format_liveness_status(result, liveness_enabled)
    print(f"❤️  Liveness: {liveness_status}")
    
    if result.get('recognized', False):
        user = result.get('user_info', {}) or {}
        print(f"\n🎉 PERSON IDENTIFIED!")
        print(f"   👤 Name: {user.get('full_name', 'Unknown')}")
        print(f"   🔢 Code: {user.get('user_code', '-')}")
        print(f"   📊 Score: {result.get('similarity_score', 0.0):.3f}")
        print(f"   🥇 Rank: #{user.get('rank', 1)}")
    else:
        print(f"\n❓ UNKNOWN PERSON")
        print(f"   📊 Best Score: {result.get('similarity_score', 0.0):.3f}")
        print(f"   💭 {UNIFORM_ACCESS_DENIED}")
    
    # Show top matches
    matches = result.get('matches') or []
    if matches:
        print(f"\n📋 TOP MATCHES:")
        print("-" * 40)
        for match in matches:
            score = match.get('similarity_score', 0.0)
            status = "✓" if score >= 0.8 else "✗"
            print(f"   {status} #{match.get('rank', 1)} {match.get('full_name', 'Unknown')} - {score:.3f}")
    
    print("=" * 50)

def display_authentication_results(result):
    """Display authentication results"""
    print("\n" + "=" * 50)
    print("       AUTHENTICATION RESULTS")
    print("=" * 50)
    
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    
    print(f"⏱️  Time: {result.get('processing_time', 0.0):.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"📊 Score: {result.get('similarity_score', 0.0):.3f}")
    
    # Use improved liveness formatting
    liveness_enabled = result.get('_liveness_enabled')
    liveness_status = format_liveness_status(result, liveness_enabled)
    print(f"❤️  Liveness: {liveness_status}")
    
    if result.get('authenticated', False):
        user = result.get('user_info', {}) or {}
        print(f"\n🎉 AUTHENTICATION SUCCESS!")
        print(f"   👤 Welcome: {user.get('full_name', 'User')}")
        print(f"   🔢 Code: {user.get('user_code', '-')}")
        print(f"   🚪 Access: ✅ GRANTED")
        print(f"   🎯 Method: Pure facial recognition")
    else:
        print(f"\n❌ {UNIFORM_ACCESS_DENIED}")
        print(f"   🚪 Access: ⛔ DENIED")
        print(f"   🎯 Method: Pure facial recognition")
    
    print("=" * 50)

def display_secure_authentication_results(result):
    """Display secure authentication results"""
    print("\n" + "=" * 50)
    print("      SECURE AUTHENTICATION RESULTS")
    print("=" * 50)
    
    if not result.get('success'):
        print(f"❌ {UNIFORM_OPERATION_FAIL}")
        return
    
    print(f"⏱️  Time: {result.get('processing_time', 0.0):.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"🛡️  Security: {result.get('security_level', 'STANDARD')}")
    print(f"📊 Score: {result.get('similarity_score', 0.0):.3f}")
    print(f"🎯 Required: {result.get('required_threshold', 0.8):.3f}")
    
    # Use improved liveness formatting (always enabled for secure)
    liveness_enabled = result.get('_liveness_enabled', True)
    liveness_status = format_liveness_status(result, liveness_enabled)
    print(f"❤️  Liveness: {liveness_status}")
    
    if result.get('authenticated', False):
        user = result.get('user_info', {}) or {}
        print(f"\n🔒 SECURE ACCESS GRANTED!")
        print(f"   👤 Authorized: {user.get('full_name', 'User')}")
        print(f"   🔢 Code: {user.get('user_code', '-')}")
        print(f"   🚪 Access: ✅ HIGH SECURITY GRANTED")
        print(f"   🎖️  Confidence: {result.get('similarity_score', 0.0):.1%}")
    else:
        print(f"\n🚫 {UNIFORM_ACCESS_DENIED}")
        print(f"   🚪 Access: ⛔ HIGH SECURITY DENIED")
        print(f"   ⚠️  {UNIFORM_ACCESS_DENIED}")
    
    print("=" * 50)

def view_statistics(recognition_system):
    """Display system statistics"""
    print("\n📊 SYSTEM STATISTICS")
    print("=" * 40)
    
    try:
        cursor = recognition_system.metadata_conn.cursor()
        
        # Total enrolled users
        cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE account_status = 'ACTIVE'")
        total_users = cursor.fetchone()[0]
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM biometric_audit_log 
            WHERE timestamp > strftime('%s', 'now', '-1 day')
        """)
        recent_activity = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN event_type LIKE '%Success' THEN 1 END) as successes,
                COUNT(*) as total
            FROM biometric_audit_log 
            WHERE timestamp > strftime('%s', 'now', '-7 days')
        """)
        week_stats = cursor.fetchone()
        
        success_rate = (week_stats[0] / week_stats[1] * 100) if week_stats[1] > 0 else 0
        
        print(f"👥 Enrolled Users: {total_users}")
        print(f"📈 24h Activity: {recent_activity} events")
        print(f"📊 7-Day Success Rate: {success_rate:.1f}%")
        print(f"🔍 Search Method: {'FAISS' if recognition_system.use_faiss else 'Linear'}")
        print(f"⚡ FAISS Index: {'✓ Active' if recognition_system.faiss_index else '✗ Not built'}")
        print(f"🎯 Similarity Threshold: {recognition_system.similarity_threshold:.3f}")
        
    except Exception:
        print(f"❌ {UNIFORM_OPERATION_FAIL}")

def main():
    """Main program with improved authentication options"""
    recognition_system = None
    
    try:
        print("🚀 Initializing Facial Recognition System...")
        recognition_system = FacialRecognitionSystem(similarity_threshold=0.8, use_faiss=True)
        print(f"✅ System ready - {len(getattr(recognition_system, 'enrolled_users', []))} users enrolled")
        
        while True:
            display_recognition_menu()
            
            choice = input("Select option (1-5): ").strip()
            
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
            else:
                print(f"❌ {UNIFORM_OPERATION_FAIL}")
            
            input("\nPress ENTER to continue...")
    
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted")
    
    except Exception:
        print(f"💥 {UNIFORM_OPERATION_FAIL}")
    
    finally:
        if recognition_system:
            recognition_system.close()
            print("🔧 System closed")

if __name__ == "__main__":
    main()