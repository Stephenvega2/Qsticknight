import logging
import sys
import traceback
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import numpy as np
import os
from quantum_visualizer import get_visualization_frames
from quantum_teleportation import QuantumTeleporter
from quantum_crypto.security_coverage import SecurityCoverage
from models import db, TeleportationResult
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # Configure SQLAlchemy with PostgreSQL
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)

    return app

# Create the Flask app
app = create_app()

# Initialize security coverage with ion trap
security_system = SecurityCoverage()

def serialize_numpy(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    return obj

@app.route('/')
def index():
    """Render the main teleportation page"""
    return render_template('teleport.html')

@app.route('/teleport', methods=['GET', 'POST'])
def teleport():
    """Handle teleportation requests with quantum security"""
    if request.method == 'POST':
        try:
            initial_state = request.form.get('initial_state', '+')
            logger.info(f"Received teleportation request for state: {initial_state}")

            # Initialize teleporter and perform teleportation
            teleporter = QuantumTeleporter(num_shots=1000)
            experience = teleporter.perform_teleportation(initial_state)

            # Store teleportation result in database
            with app.app_context():
                # Convert numpy values to Python types for JSON storage
                validation_metrics = serialize_numpy(experience['metrics']['validation'])
                tomography_results = serialize_numpy(experience.get('tomography', {}))

                result = TeleportationResult(
                    initial_state=initial_state,
                    success_probability=float(experience['metrics']['success_probability']),
                    entropy=float(experience['metrics']['entropy']),
                    channel_security=float(experience['metrics']['channel_security']),
                    circuit_depth=experience['circuit_depth'],
                    validation_metrics=validation_metrics,
                    tomography_results=tomography_results
                )
                db.session.add(result)
                db.session.commit()
                logger.info(f"Saved teleportation result with ID {result.id}")

            # Convert numpy types to native Python types for template
            experience = serialize_numpy(experience)

            return render_template('teleport.html', 
                                  experience=experience,
                                  initial_state=initial_state)
        except Exception as e:
            logger.error(f"Teleportation error: {str(e)}")
            traceback.print_exc()
            return render_template('teleport.html', error=str(e))

    return render_template('teleport.html')

@app.route('/quantum_states')
def quantum_states():
    """Display quantum states and security metrics"""
    try:
        with app.app_context():
            # Get all stored quantum states
            results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(10).all()

            # Calculate aggregate metrics
            total_results = len(results)
            if total_results > 0:
                avg_success = sum(r.success_probability for r in results) / total_results
                avg_entropy = sum(r.entropy for r in results) / total_results
                avg_security = sum(r.channel_security for r in results) / total_results
            else:
                avg_success = avg_entropy = avg_security = 0

            # Format metrics for display
            metrics = {
                'total_states': total_results,
                'avg_success': avg_success,
                'avg_entropy': avg_entropy,
                'avg_security': avg_security
            }

            return render_template('quantum_states.html',
                                   results=results,
                                   metrics=metrics)
    except Exception as e:
        logger.error(f"Error analyzing quantum states: {str(e)}")
        traceback.print_exc()
        return render_template('quantum_states.html', 
                               results=[],
                               metrics={'total_states': 0, 'avg_success': 0, 'avg_entropy': 0, 'avg_security': 0})


@app.route('/quantum-visualization')
def quantum_visualization():
    """Display live quantum visualization"""
    try:
        # Import visualization frames
        frames = get_visualization_frames()

        with app.app_context():
            # Get latest results for metrics
            latest_results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(5).all()

            # Calculate metrics
            metrics = {
                'success_rate': sum(r.success_probability for r in latest_results) / len(latest_results) if latest_results else 0,
                'entropy': sum(r.entropy for r in latest_results) / len(latest_results) if latest_results else 0,
                'security': sum(r.channel_security for r in latest_results) / len(latest_results) if latest_results else 0
            }

            return render_template('quantum_visualization.html',
                                frames=frames,
                                metrics=metrics)
    except Exception as e:
        logger.error(f"Error loading quantum visualization: {str(e)}")
        traceback.print_exc()
        return render_template('quantum_visualization.html', error=str(e))

@app.route('/roadmap')
def roadmap():
    """Display project roadmap and current progress"""
    try:
        # Get stats for the roadmap
        with app.app_context():
            total_teleportations = TeleportationResult.query.count()
            recent_results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(5).all()

            avg_success = 0
            avg_security = 0
            if recent_results:
                avg_success = sum(r.success_probability for r in recent_results) / len(recent_results)
                avg_security = sum(r.channel_security for r in recent_results) / len(recent_results)

            stats = {
                'active_users': len(set(r.user_id for r in recent_results if r.user_id)) if recent_results else 0,
                'states_captured': total_teleportations,
                'security_score': int(avg_security * 100),
                'total_rewards': int(avg_success * total_teleportations)
            }

            return render_template('roadmap.html', stats=stats)
    except Exception as e:
        logger.error(f"Error loading roadmap: {str(e)}")
        traceback.print_exc()
        return render_template('roadmap.html', error=str(e))

@app.route('/network_stats')
def network_stats():
    """API endpoint for network statistics"""
    try:
        with app.app_context():
            total_teleportations = TeleportationResult.query.count()
            recent_results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(5).all()

            avg_success = 0
            avg_security = 0
            if recent_results:
                avg_success = sum(r.success_probability for r in recent_results) / len(recent_results)
                avg_security = sum(r.channel_security for r in recent_results) / len(recent_results)

            return jsonify({
                'active_users': len(set(r.user_id for r in recent_results if r.user_id)) if recent_results else 0,
                'states_captured': total_teleportations,
                'security_score': int(avg_security * 100),
                'total_rewards': int(avg_success * total_teleportations)
            })
    except Exception as e:
        logger.error(f"Error getting network stats: {str(e)}")
        return jsonify({
            'active_users': 0,
            'states_captured': 0,
            'security_score': 0,
            'total_rewards': 0
        })

@app.route('/security-dashboard')
def security_dashboard():
    """Display the security dashboard with ion trap protection status"""
    try:
        # Get device and security information
        with app.app_context():
            results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(5).all()

            # Calculate security metrics
            avg_success = 0
            avg_security = 0
            if results:
                avg_success = sum(r.success_probability for r in results) / len(results)
                avg_security = sum(r.channel_security for r in results) / len(results)

            # Get current user's device protection status
            hostname = socket.gethostname()
            try:
                ip_addr = socket.gethostbyname(hostname)
                device_stability = 0.95  # Example value
            except:
                ip_addr = "unknown"
                device_stability = 0.0

            # Get security validation status
            security_status = {
                'protection_status': 'Active' if device_stability > 0.7 else 'At Risk',
                'ion_trap_active': device_stability > 0.85,
                'device_stability': device_stability,
                'validation_bonus': 1.2 if device_stability > 0.9 else 1.0,
                'security_score': avg_security,
                'reward_points': int(avg_success * 1000),
                'total_rewards': int(avg_success * len(results) * 100)
            }

            return render_template('security_dashboard.html', security=security_status)

    except Exception as e:
        logger.error(f"Error loading security dashboard: {str(e)}")
        traceback.print_exc()
        return render_template('security_dashboard.html', error=str(e))

@app.route('/audit/api/security-status')
def security_status():
    """API endpoint for real-time security metrics"""
    try:
        with app.app_context():
            results = TeleportationResult.query.order_by(TeleportationResult.timestamp.desc()).limit(5).all()

            # Calculate current security metrics
            if results:
                avg_success = sum(r.success_probability for r in results) / len(results)
                avg_security = sum(r.channel_security for r in results) / len(results)
            else:
                avg_success = avg_security = 0

            # Get device stability
            try:
                hostname = socket.gethostname()
                ip_addr = socket.gethostbyname(hostname)
                device_stability = 0.95  # Example value
            except:
                device_stability = 0.0

            return jsonify({
                'protection_status': 'Active' if device_stability > 0.7 else 'At Risk',
                'ion_trap_active': device_stability > 0.85,
                'device_stability': device_stability,
                'validation_bonus': 1.2 if device_stability > 0.9 else 1.0,
                'security_score': avg_security,
                'reward_points': int(avg_success * 1000),
                'total_rewards': int(avg_success * len(results) * 100)
            })
    except Exception as e:
        logger.error(f"Error getting security status: {str(e)}")
        return jsonify({
            'error': str(e),
            'protection_status': 'Unknown',
            'ion_trap_active': False,
            'device_stability': 0,
            'validation_bonus': 1.0,
            'security_score': 0,
            'reward_points': 0,
            'total_rewards': 0
        })

@app.route('/live-monitor')
def live_monitor():
    """Display live monitoring of quantum operations and security checks"""
    try:
        with app.app_context():
            # Get latest teleportation results for initial state
            latest_results = TeleportationResult.query.order_by(
                TeleportationResult.timestamp.desc()
            ).limit(10).all()

            # Calculate real-time metrics
            metrics = {
                'teleportations': len(latest_results),
                'avg_success': sum(r.success_probability for r in latest_results) / len(latest_results) if latest_results else 0,
                'avg_security': sum(r.channel_security for r in latest_results) / len(latest_results) if latest_results else 0,
                'avg_entropy': sum(r.entropy for r in latest_results) / len(latest_results) if latest_results else 0
            }

            return render_template('live_monitor.html', metrics=metrics)
    except Exception as e:
        logger.error(f"Error loading live monitor: {str(e)}")
        traceback.print_exc()
        return render_template('live_monitor.html', error=str(e))

@app.route('/api/live-metrics')
def live_metrics():
    """API endpoint for real-time quantum metrics"""
    try:
        with app.app_context():
            # Get latest result for real-time data
            latest = TeleportationResult.query.order_by(
                TeleportationResult.timestamp.desc()
            ).first()

            if latest:
                # Get device information
                hostname = socket.gethostname()
                try:
                    ip_addr = socket.gethostbyname(hostname)
                    device_stability = 0.95
                except:
                    ip_addr = "unknown"
                    device_stability = 0.0

                return jsonify({
                    'timestamp': latest.timestamp.isoformat(),
                    'success_rate': float(latest.success_probability),
                    'entropy': float(latest.entropy),
                    'channel_security': float(latest.channel_security),
                    'device_protection': {
                        'stability': device_stability,
                        'ion_trap_active': device_stability > 0.85,
                        'protection_level': 'Active' if device_stability > 0.7 else 'At Risk'
                    },
                    'validation_metrics': latest.validation_metrics
                })

            return jsonify({
                'error': 'No data available'
            })
    except Exception as e:
        logger.error(f"Error getting live metrics: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        # Create database tables
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")

        logger.info("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug to False
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        traceback.print_exc()
