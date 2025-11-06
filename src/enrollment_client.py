import sys
import re
from enrollment_controller import UserEnrollmentSystem

def validate_user_code(user_code: str) -> bool:
    """Validate 10-digit user code format"""
    return bool(re.match(r'^\d{10}$', user_code))

def validate_name(name: str) -> bool:
    """Validate name contains only letters and spaces"""
    return bool(re.match(r'^[a-zA-Z\s]+$', name)) and len(name.strip()) > 0

def get_user_input():
    """Get user enrollment information"""
    print("\n" + "=" * 50)
    print("    FACIAL RECOGNITION USER ENROLLMENT")
    print("=" * 50)
    
    # Get user code
    while True:
        user_code = input("Enter 10-digit user code: ").strip()
        if validate_user_code(user_code):
            break
        print("❌ ERROR: User code must be exactly 10 digits")
    
    # Get first name
    while True:
        first_name = input("Enter first name: ").strip()
        if validate_name(first_name):
            break
        print("❌ ERROR: First name must contain only letters")
    
    # Get last name
    while True:
        last_name = input("Enter last name: ").strip()
        if validate_name(last_name):
            break
        print("❌ ERROR: Last name must contain only letters")
    
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
            print("❌ ERROR: Please enter Y or N")
    
    return {
        'user_code': user_code,
        'first_name': first_name.title(),
        'last_name': last_name.title(),
        'enable_liveness': enable_liveness
    }

def confirm_enrollment(user_info):
    """Display confirmation and get approval"""
    print("\n" + "=" * 30)
    print("   ENROLLMENT CONFIRMATION")
    print("=" * 30)
    print(f"👤 User Code: {user_info['user_code']}")
    print(f"📝 Name: {user_info['first_name']} {user_info['last_name']}")
    print(f"❤️  Liveness: {'✅ Enabled' if user_info['enable_liveness'] else '❌ Disabled'}")
    print("=" * 30)
    
    confirm = input("\nProceed with enrollment? (Y/n) [Y]: ").strip().lower()
    return confirm in ['', 'y', 'yes']

def main():
    """Main enrollment program"""
    enrollment_system = None
    
    try:
        # Get user input
        user_info = get_user_input()
        
        # Confirm enrollment
        if not confirm_enrollment(user_info):
            print("❌ Enrollment cancelled by user")
            return
        
        print("\n🚀 Initializing enrollment system...")
        
        # Initialize system  
        enrollment_system = UserEnrollmentSystem()
        
        # Perform enrollment
        print("📷 Prepare for face capture...")
        print("   Position your face in the camera frame")
        print("   Press SPACE when ready, ESC to cancel")
        
        result = enrollment_system.enroll_new_user(
            user_code=user_info['user_code'],
            first_name=user_info['first_name'],
            last_name=user_info['last_name'],
            enable_liveness=user_info['enable_liveness']
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("         ENROLLMENT RESULTS")
        print("=" * 50)
        
        if result['success']:
            print("🎉 SUCCESS: User enrolled successfully!")
            print(f"   👤 User: {user_info['first_name']} {user_info['last_name']}")
            print(f"   🔢 Code: {user_info['user_code']}")
            print(f"   📊 Score: {result['similarity_score']:.3f}")
            print(f"   🆔 ID: {result['user_id']}")
        else:
            print("❌ FAILED: Enrollment unsuccessful")
            print(f"   💬 Reason: {result['message']}")
            print(f"   📊 Score: {result['similarity_score']:.3f}")
        
        print("=" * 50)
            
    except KeyboardInterrupt:
        print("\n⚠️  Enrollment cancelled by user")
        
    except Exception as e:
        print(f"\n💥 SYSTEM ERROR: {e}")
        
    finally:
        if enrollment_system:
            enrollment_system.close()
            print("🔧 System closed")

if __name__ == "__main__":
    main()