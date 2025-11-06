import sys
import cv2
import numpy as np
from recognition_controller import FacialRecognitionSystem

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

def recognize_from_camera(recognition_system):
    """Handle camera-based face identification"""
    print("\n📷 FACE IDENTIFICATION MODE")
    print("=" * 40)
    
    # Get liveness preference
    while True:
        liveness_choice = input("Enable liveness detection? (Y/n) [Y]: ").strip().lower()
        if liveness_choice in ['', 'y', 'yes']:
            enable_liveness = True
            break
        elif liveness_choice in ['n', 'no']:
            enable_liveness = False
            break
        else:
            print("❌ Please enter 'Y' or 'N'")
    
    # Get number of matches to show
    while True:
        try:
            top_k = input("Top matches to show (1-10) [5]: ").strip()
            if top_k == '':
                top_k = 5
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
        display_recognition_results(result)
        
    except KeyboardInterrupt:
        print("\n⚠️  Recognition cancelled")

def authenticate_user_pure(recognition_system):
    """Pure facial authentication - no user code required"""
    print("\n🔐 FACIAL AUTHENTICATION")
    print("=" * 40)
    print("📷 Look at camera to authenticate - No ID required!")
    
    # Get liveness preference
    while True:
        liveness_choice = input("Enable liveness detection? (Y/n) [Y]: ").strip().lower()
        if liveness_choice in ['', 'y', 'yes']:
            enable_liveness = True
            break
        elif liveness_choice in ['n', 'no']:
            enable_liveness = False
            break
        else:
            print("❌ Please enter 'Y' or 'N'")
    
    print(f"\n🎯 Authenticating via facial recognition...")
    print(f"   Liveness: {'✓ ON' if enable_liveness else '✗ OFF'}")
    print("📷 Position face in camera and press SPACE")
    
    try:
        result = recognition_system.recognize_from_camera(enable_liveness, top_k=1)
        
        if result['success'] and result['recognized']:
            # User authenticated successfully
            user = result['user_info']
            auth_result = {
                'success': True,
                'authenticated': True,
                'user_info': user,
                'similarity_score': result['similarity_score'],
                'liveness_result': result['liveness_result'],
                'processing_time': result['processing_time'],
                'search_method': result.get('search_method', 'Unknown'),
                'message': f"Welcome {user['full_name']}!"
            }
        else:
            # Authentication failed
            auth_result = {
                'success': True,
                'authenticated': False,
                'user_info': None,
                'similarity_score': result.get('similarity_score', 0.0),
                'liveness_result': result.get('liveness_result', {'is_live': False}),
                'processing_time': result.get('processing_time', 0.0),
                'search_method': result.get('search_method', 'Unknown'),
                'message': 'Face not recognized - Access denied'
            }
        
        display_authentication_results(auth_result)
        
    except KeyboardInterrupt:
        print("\n⚠️  Authentication cancelled")

def authenticate_secure_access(recognition_system):
    """High-security authentication with elevated threshold"""
    print("\n🔒 SECURE ACCESS AUTHENTICATION")
    print("=" * 40)
    print("🛡️  High security mode - Higher confidence required")
    
    # Get liveness preference (forced ON for secure access)
    print("🔐 Liveness detection: ✓ MANDATORY for secure access")
    enable_liveness = True
    
    # Set high security threshold
    high_security_threshold = 0.85
    
    print(f"\n🎯 Secure authentication starting...")
    print(f"   Confidence required: {high_security_threshold:.1%}")
    print(f"   Liveness: ✓ MANDATORY")
    print("📷 Position face in camera and press SPACE")
    
    try:
        # Temporarily adjust threshold for high security
        original_threshold = recognition_system.similarity_threshold
        recognition_system.similarity_threshold = high_security_threshold
        
        result = recognition_system.recognize_from_camera(enable_liveness=True, top_k=1)
        
        # Restore original threshold
        recognition_system.similarity_threshold = original_threshold
        
        if result['success'] and result['recognized']:
            user = result['user_info']
            auth_result = {
                'success': True,
                'authenticated': True,
                'user_info': user,
                'similarity_score': result['similarity_score'],
                'liveness_result': result['liveness_result'],
                'processing_time': result['processing_time'],
                'search_method': result.get('search_method', 'Unknown'),
                'security_level': 'HIGH',
                'required_threshold': high_security_threshold,
                'message': f"Secure access granted to {user['full_name']}"
            }
        else:
            auth_result = {
                'success': True,
                'authenticated': False,
                'user_info': None,
                'similarity_score': result.get('similarity_score', 0.0),
                'liveness_result': result.get('liveness_result', {'is_live': False}),
                'processing_time': result.get('processing_time', 0.0),
                'search_method': result.get('search_method', 'Unknown'),
                'security_level': 'HIGH',
                'required_threshold': high_security_threshold,
                'message': f'Secure access denied - Required: {high_security_threshold:.3f}, Got: {result.get("similarity_score", 0):.3f}'
            }
        
        display_secure_authentication_results(auth_result)
        
    except KeyboardInterrupt:
        print("\n⚠️  Secure authentication cancelled")

def display_recognition_results(result):
    """Display identification results"""
    print("\n" + "=" * 50)
    print("         IDENTIFICATION RESULTS")
    print("=" * 50)
    
    if not result['success']:
        print(f"❌ FAILED: {result['message']}")
        return
    
    print(f"⏱️  Time: {result['processing_time']:.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"👥 Database: {result.get('database_size', 0)} users")
    print(f"❤️  Liveness: {'✓ PASS' if result['liveness_result']['is_live'] else '✗ FAIL'}")
    
    if result['recognized']:
        user = result['user_info']
        print(f"\n🎉 PERSON IDENTIFIED!")
        print(f"   👤 Name: {user['full_name']}")
        print(f"   🔢 Code: {user['user_code']}")
        print(f"   📊 Score: {result['similarity_score']:.3f}")
        print(f"   🥇 Rank: #{user['rank']}")
    else:
        print(f"\n❓ UNKNOWN PERSON")
        print(f"   📊 Best Score: {result['similarity_score']:.3f}")
        print(f"   💭 Person not in database")
    
    # Show top matches
    if result['matches']:
        print(f"\n📋 TOP MATCHES:")
        print("-" * 40)
        for match in result['matches']:
            status = "✓" if match['similarity_score'] >= 0.8 else "✗"
            print(f"   {status} #{match['rank']} {match['full_name']} - {match['similarity_score']:.3f}")
    
    print("=" * 50)

def display_authentication_results(result):
    """Display authentication results"""
    print("\n" + "=" * 50)
    print("       AUTHENTICATION RESULTS")
    print("=" * 50)
    
    if not result['success']:
        print(f"❌ ERROR: {result['message']}")
        return
    
    print(f"⏱️  Time: {result['processing_time']:.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"📊 Score: {result['similarity_score']:.3f}")
    print(f"❤️  Liveness: {'✓ PASS' if result['liveness_result']['is_live'] else '✗ FAIL'}")
    
    if result['authenticated']:
        user = result['user_info']
        print(f"\n🎉 AUTHENTICATION SUCCESS!")
        print(f"   👤 Welcome: {user['full_name']}")
        print(f"   🔢 Code: {user['user_code']}")
        print(f"   🚪 Access: ✅ GRANTED")
        print(f"   🎯 Method: Pure facial recognition")
    else:
        print(f"\n❌ AUTHENTICATION FAILED!")
        print(f"   🚪 Access: ⛔ DENIED")
        print(f"   💬 Reason: {result['message']}")
        print(f"   🎯 Method: Pure facial recognition")
    
    print("=" * 50)

def display_secure_authentication_results(result):
    """Display secure authentication results"""
    print("\n" + "=" * 50)
    print("      SECURE AUTHENTICATION RESULTS")
    print("=" * 50)
    
    if not result['success']:
        print(f"❌ ERROR: {result['message']}")
        return
    
    print(f"⏱️  Time: {result['processing_time']:.3f}s")
    print(f"🔍 Method: {result.get('search_method', 'Unknown')}")
    print(f"🛡️  Security: {result.get('security_level', 'STANDARD')}")
    print(f"📊 Score: {result['similarity_score']:.3f}")
    print(f"🎯 Required: {result.get('required_threshold', 0.8):.3f}")
    print(f"❤️  Liveness: {'✓ PASS' if result['liveness_result']['is_live'] else '✗ FAIL'}")
    
    if result['authenticated']:
        user = result['user_info']
        print(f"\n🔒 SECURE ACCESS GRANTED!")
        print(f"   👤 Authorized: {user['full_name']}")
        print(f"   🔢 Code: {user['user_code']}")
        print(f"   🚪 Access: ✅ HIGH SECURITY GRANTED")
        print(f"   🎖️  Confidence: {result['similarity_score']:.1%}")
    else:
        print(f"\n🚫 SECURE ACCESS DENIED!")
        print(f"   🚪 Access: ⛔ HIGH SECURITY DENIED")
        print(f"   💬 Reason: {result['message']}")
        print(f"   ⚠️  Insufficient confidence for secure access")
    
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
        
    except Exception as e:
        print(f"❌ Error loading stats: {e}")

def main():
    """Main program with improved authentication options"""
    recognition_system = None
    
    try:
        print("🚀 Initializing Facial Recognition System...")
        recognition_system = FacialRecognitionSystem(similarity_threshold=0.8, use_faiss=True)
        print(f"✅ System ready - {len(recognition_system.enrolled_users)} users enrolled")
        
        while True:
            display_recognition_menu()
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                recognize_from_camera(recognition_system)
            elif choice == '2':
                authenticate_user_pure(recognition_system)  # NEW: Pure face authentication
            elif choice == '3':
                authenticate_secure_access(recognition_system)  # NEW: High security authentication
            elif choice == '4':
                view_statistics(recognition_system)
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
            
            input("\nPress ENTER to continue...")
    
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted")
    
    except Exception as e:
        print(f"💥 ERROR: {e}")
    
    finally:
        if recognition_system:
            recognition_system.close()
            print("🔧 System closed")

if __name__ == "__main__":
    main()