import getpass
import sys
import re
import logging
from enrollment_controller import UserEnrollmentSystem
from rbac import RBACManager, ROLE_ADMIN

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure structured logging for enrollment."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("facial_recognition.log", encoding="utf-8"),
        ],
    )


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

    while True:
        user_code = input("Enter 10-digit user code: ").strip()
        if validate_user_code(user_code):
            break
        print("ERROR: User code must be exactly 10 digits")

    while True:
        first_name = input("Enter first name: ").strip()
        if validate_name(first_name):
            break
        print("ERROR: First name must contain only letters")

    while True:
        last_name = input("Enter last name: ").strip()
        if validate_name(last_name):
            break
        print("ERROR: Last name must contain only letters")

    while True:
        liveness_choice = input("Enable liveness detection? (Y/n) [Y]: ").strip().lower()
        if liveness_choice in ['', 'y', 'yes']:
            enable_liveness = True
            break
        elif liveness_choice in ['n', 'no']:
            enable_liveness = False
            break
        else:
            print("ERROR: Please enter Y or N")

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
    print(f"  User Code: {user_info['user_code']}")
    print(f"  Name: {user_info['first_name']} {user_info['last_name']}")
    print(f"  Liveness: {'Enabled' if user_info['enable_liveness'] else 'Disabled'}")
    print("=" * 30)

    confirm = input("\nProceed with enrollment? (Y/n) [Y]: ").strip().lower()
    return confirm in ['', 'y', 'yes']


def authenticate_operator() -> bool:
    """Require operator authentication before enrollment."""
    rbac = RBACManager()

    if not rbac.has_operators():
        print("\nNo operator accounts found. Creating initial admin account.")
        op_id = input("Enter operator ID: ").strip()
        if not op_id:
            return False
        pin = getpass.getpass("Enter PIN: ")
        pin_confirm = getpass.getpass("Confirm PIN: ")
        if pin != pin_confirm:
            print("PINs do not match")
            return False
        if len(pin) < 4:
            print("PIN must be at least 4 characters")
            return False
        if not rbac.register_operator(op_id, pin, role=ROLE_ADMIN):
            print("Failed to create account")
            return False
        print(f"Admin account '{op_id}' created.")

    print("\nOPERATOR AUTHENTICATION REQUIRED")
    print("=" * 35)
    op_id = input("Operator ID: ").strip()
    if not op_id:
        return False
    pin = getpass.getpass("PIN: ")

    if not rbac.authenticate_operator(op_id, pin):
        print("Authentication failed")
        return False

    if not rbac.check_permission("enroll"):
        print(f"Permission denied: role '{rbac.current_role}' cannot perform enrollment")
        return False

    print(f"Authenticated as {op_id} ({rbac.current_role})")
    return True


def main():
    """Main enrollment program"""
    setup_logging()
    enrollment_system = None

    try:
        # Require operator authentication
        if not authenticate_operator():
            print("Enrollment requires operator authentication. Exiting.")
            return

        user_info = get_user_input()

        if not confirm_enrollment(user_info):
            print("Enrollment cancelled by user")
            return

        logger.info("Initializing enrollment system...")
        enrollment_system = UserEnrollmentSystem()

        print("\nPrepare for face capture...")
        print("   Position your face in the camera frame")
        print("   Press SPACE when ready, ESC to cancel")

        result = enrollment_system.enroll_new_user(
            user_code=user_info['user_code'],
            first_name=user_info['first_name'],
            last_name=user_info['last_name'],
            enable_liveness=user_info['enable_liveness']
        )

        print("\n" + "=" * 50)
        print("         ENROLLMENT RESULTS")
        print("=" * 50)

        if result['success']:
            logger.info("Enrollment succeeded for %s %s (code=%s)",
                        user_info['first_name'], user_info['last_name'], user_info['user_code'])
            print("SUCCESS: User enrolled successfully!")
            print(f"   User: {user_info['first_name']} {user_info['last_name']}")
            print(f"   Code: {user_info['user_code']}")
            print(f"   Score: {result['similarity_score']:.3f}")
            print(f"   ID: {result['user_id']}")
        else:
            logger.warning("Enrollment failed for code=%s: %s",
                          user_info['user_code'], result['message'])
            print("FAILED: Enrollment unsuccessful")
            print(f"   Reason: {result['message']}")
            print(f"   Score: {result['similarity_score']:.3f}")

        print("=" * 50)

    except KeyboardInterrupt:
        logger.info("Enrollment cancelled by user (KeyboardInterrupt)")
        print("\nEnrollment cancelled by user")

    except Exception as e:
        logger.exception("System error during enrollment")
        print(f"\nSYSTEM ERROR: {e}")

    finally:
        if enrollment_system:
            enrollment_system.close()


if __name__ == "__main__":
    main()
