from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
import pyotp
import qrcode
import io
import base64
from models import User, UserStore
import os
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint('auth', __name__)
user_store = UserStore()

# OAuth configuration
oauth = OAuth()

# Google OAuth
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# GitHub OAuth
github = oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)


def init_auth(app):
    """Initialize authentication for the Flask app"""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    oauth.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return user_store.get_user(user_id)
    
    return login_manager


@auth_bp.route('/login')
def login():
    """Display login page"""
    return render_template('login.html')


@auth_bp.route('/login/google')
def google_login():
    """Initiate Google OAuth login"""
    redirect_uri = url_for('auth.google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)


@auth_bp.route('/authorize/google')
def google_authorize():
    """Handle Google OAuth callback"""
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if user_info:
            email = user_info['email']
            name = user_info.get('name', email)
            
            # Get or create user
            user = user_store.get_user_by_email(email)
            if not user:
                user = user_store.create_user(email, name, 'google')
            
            # Store user in session for 2FA verification
            session['pending_user_id'] = user.id
            
            # Check if 2FA is enabled
            if user.two_factor_secret:
                return redirect(url_for('auth.verify_2fa'))
            else:
                # First time login, set up 2FA
                return redirect(url_for('auth.setup_2fa'))
        
        flash('Failed to get user information from Google', 'error')
        return redirect(url_for('auth.login'))
    
    except Exception as e:
        flash(f'Authentication failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))


@auth_bp.route('/login/github')
def github_login():
    """Initiate GitHub OAuth login"""
    redirect_uri = url_for('auth.github_authorize', _external=True)
    return github.authorize_redirect(redirect_uri)


@auth_bp.route('/authorize/github')
def github_authorize():
    """Handle GitHub OAuth callback"""
    try:
        token = github.authorize_access_token()
        
        # Get user info
        resp = github.get('user', token=token)
        user_info = resp.json()
        
        # Get email (might need separate call if primary email is private)
        email = user_info.get('email')
        if not email:
            emails_resp = github.get('user/emails', token=token)
            emails = emails_resp.json()
            # Get primary email
            for e in emails:
                if e.get('primary'):
                    email = e.get('email')
                    break
        
        if email:
            name = user_info.get('name') or user_info.get('login')
            
            # Get or create user
            user = user_store.get_user_by_email(email)
            if not user:
                user = user_store.create_user(email, name, 'github')
            
            # Store user in session for 2FA verification
            session['pending_user_id'] = user.id
            
            # Check if 2FA is enabled
            if user.two_factor_secret:
                return redirect(url_for('auth.verify_2fa'))
            else:
                # First time login, set up 2FA
                return redirect(url_for('auth.setup_2fa'))
        
        flash('Failed to get user information from GitHub', 'error')
        return redirect(url_for('auth.login'))
    
    except Exception as e:
        flash(f'Authentication failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))


@auth_bp.route('/setup-2fa')
def setup_2fa():
    """Set up 2FA for the user"""
    if 'pending_user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user = user_store.get_user(session['pending_user_id'])
    if not user:
        return redirect(url_for('auth.login'))
    
    # Generate new secret if not exists
    if not user.two_factor_secret:
        secret = pyotp.random_base32()
        user.two_factor_secret = secret
        user_store.update_user(user)
    
    # Generate QR code
    totp = pyotp.TOTP(user.two_factor_secret)
    provisioning_uri = totp.provisioning_uri(
        name=user.email,
        issuer_name='Nutritional Insights App'
    )
    
    # Create QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    qr_code_data = base64.b64encode(buffer.getvalue()).decode()
    
    return render_template('setup_2fa.html', 
                         qr_code=qr_code_data, 
                         secret=user.two_factor_secret,
                         user=user)


@auth_bp.route('/verify-2fa', methods=['GET', 'POST'])
def verify_2fa():
    """Verify 2FA code"""
    if 'pending_user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user = user_store.get_user(session['pending_user_id'])
    if not user or not user.two_factor_secret:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        
        # Verify the code
        totp = pyotp.TOTP(user.two_factor_secret)
        if totp.verify(code, valid_window=1):
            # Code is valid, log in the user
            login_user(user)
            session.pop('pending_user_id', None)
            flash(f'Welcome, {user.name}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid 2FA code. Please try again.', 'error')
    
    return render_template('verify_2fa.html', user=user)


@auth_bp.route('/logout')
@login_required
def logout():
    """Log out the current user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/security-status')
@login_required
def security_status():
    """Show security and compliance status"""
    return render_template('security_status.html', user=current_user)

@auth_bp.route('/verify-2fa-test', methods=['POST'])
@login_required
def verify_2fa_test():
    """Test endpoint to verify 2FA codes"""
    import pyotp
    from flask import request, jsonify
    
    data = request.get_json()
    code = data.get('code', '')
    
    if not current_user.two_factor_secret:
        return jsonify({'valid': False, 'error': '2FA not enabled'})
    
    totp = pyotp.TOTP(current_user.two_factor_secret)
    is_valid = totp.verify(code, valid_window=1)
    
    return jsonify({'valid': is_valid})