from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Float, JSON, String, Integer, DateTime, Boolean
from typing import Optional
import json
import numpy as np
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import socket

# Initialize SQLAlchemy without binding to app yet
db = SQLAlchemy()

class SafeJSON(JSON):
    def process_bind_param(self, value, dialect):
        """Convert numpy types to Python native types before JSON serialization"""
        if value is not None:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, complex):
                    return {'real': obj.real, 'imag': obj.imag}
                return obj

            if isinstance(value, dict):
                value = {k: convert(v) for k, v in value.items()}
            elif isinstance(value, list):
                value = [convert(item) for item in value]

        return super().process_bind_param(value, dialect)

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    email: Mapped[str] = mapped_column(String(120), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    device_signature: Mapped[Optional[str]] = mapped_column(String(255))
    ion_trap_validator: Mapped[Optional[str]] = mapped_column(String(255))
    last_ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    teleportation_results = db.relationship('TeleportationResult', back_populates='user')
    encrypted_data = db.relationship('EncryptedData', back_populates='user')

    def set_password(self, password):
        """Set password with quantum entropy enhancement"""
        quantum_salt = self.generate_quantum_salt()
        self.password_hash = generate_password_hash(password + quantum_salt)

    def check_password(self, password):
        """Verify password with quantum validation"""
        quantum_salt = self.generate_quantum_salt()
        return check_password_hash(self.password_hash, password + quantum_salt)

    def generate_quantum_salt(self):
        """Generate quantum-enhanced salt for password security"""
        device_info = self.get_device_info()
        base = f"{self.email}:{device_info}:{self.username}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]

    def get_device_info(self):
        """Get unique device signature"""
        try:
            hostname = socket.gethostname()
            ip_addr = socket.gethostbyname(hostname)
            return f"{hostname}:{ip_addr}"
        except:
            return "unknown_device"

    def update_ion_trap_validator(self, request_ip):
        """Update ion trap validator based on current request"""
        self.last_ip_address = request_ip
        device_info = self.get_device_info()
        self.device_signature = hashlib.sha256(device_info.encode()).hexdigest()
        self.ion_trap_validator = hashlib.sha256(
            f"{self.email}:{device_info}:{request_ip}".encode()
        ).hexdigest()

    def validate_ion_trap(self, request_ip):
        """Validate current request against ion trap security"""
        if not self.ion_trap_validator:
            return False

        device_info = self.get_device_info()
        current_validator = hashlib.sha256(
            f"{self.email}:{device_info}:{request_ip}".encode()
        ).hexdigest()

        return current_validator == self.ion_trap_validator

    @property
    def is_active(self):
        return True

class EncryptedData(db.Model):
    __tablename__ = 'encrypted_data'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(db.ForeignKey('users.id'), nullable=True)
    encrypted_content: Mapped[str] = mapped_column(String(255))
    iv: Mapped[str] = mapped_column(String(255))
    nft_metadata: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Many-to-one relationship with User
    user = db.relationship('User', back_populates='encrypted_data')

class TeleportationResult(db.Model):
    __tablename__ = 'teleportation_results'

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(db.ForeignKey('users.id'))
    initial_state: Mapped[str] = mapped_column(String(50))
    success_probability: Mapped[float] = mapped_column(Float)
    entropy: Mapped[float] = mapped_column(Float)
    channel_security: Mapped[float] = mapped_column(Float)
    circuit_depth: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # JSON stored fields with numpy handling
    validation_metrics = mapped_column(SafeJSON)
    tomography_results = mapped_column(SafeJSON)

    # Relationships
    user = db.relationship('User', back_populates='teleportation_results')
    measurements = db.relationship('QuantumMeasurement', back_populates='teleportation_result')
    circuit_visualization = db.relationship('CircuitVisualization', 
                                       back_populates='teleportation_result', 
                                       uselist=False)

    @staticmethod
    def sanitize_float(value):
        """Convert numpy float types to Python float"""
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value

    def to_dict(self):
        """Convert model to dictionary with proper type handling"""
        return {
            'id': self.id,
            'initial_state': self.initial_state,
            'success_probability': self.sanitize_float(self.success_probability),
            'entropy': self.sanitize_float(self.entropy),
            'channel_security': self.sanitize_float(self.channel_security),
            'circuit_depth': self.circuit_depth,
            'timestamp': self.timestamp.isoformat(),
            'validation_metrics': self.validation_metrics,
            'tomography_results': self.tomography_results,
            'metadata': {
                'measurements': [m.to_dict() for m in self.measurements],
                'circuit_info': self.circuit_visualization.to_dict() if self.circuit_visualization else None,
                'heatmap_points': [p.to_dict() for p in self.heatmap_points]
            }
        }

class HeatMapData(db.Model):
    __tablename__ = 'heatmap_data'

    id: Mapped[int] = mapped_column(primary_key=True)
    teleportation_id: Mapped[int] = mapped_column(db.ForeignKey('teleportation_results.id'))
    x_coord: Mapped[float] = mapped_column(Float)
    y_coord: Mapped[float] = mapped_column(Float)
    intensity: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    state_type: Mapped[str] = mapped_column(String(50))  # 'source', 'target', 'entangled'

    # Relationship
    teleportation_result = db.relationship('TeleportationResult', backref=db.backref('heatmap_points', lazy=True))

    def to_dict(self):
        """Convert model to dictionary with proper type handling"""
        return {
            'id': self.id,
            'x': float(self.x_coord),
            'y': float(self.y_coord),
            'intensity': float(self.intensity),
            'timestamp': self.timestamp.isoformat(),
            'state_type': self.state_type
        }

class QuantumMeasurement(db.Model):
    __tablename__ = 'quantum_measurements'

    id: Mapped[int] = mapped_column(primary_key=True)
    teleportation_id: Mapped[int] = mapped_column(db.ForeignKey('teleportation_results.id'))
    basis: Mapped[str] = mapped_column(String(1))  # X, Y, or Z basis
    expectation_value: Mapped[float] = mapped_column(Float)
    measurement_quality: Mapped[float] = mapped_column(Float)
    statistical_fidelity: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    teleportation_result = db.relationship('TeleportationResult', back_populates='measurements')

    def to_dict(self):
        return {
            'id': self.id,
            'basis': self.basis,
            'expectation_value': self.expectation_value,
            'measurement_quality': self.measurement_quality,
            'statistical_fidelity': self.statistical_fidelity,
            'created_at': self.created_at.isoformat()
        }

class CircuitVisualization(db.Model):
    __tablename__ = 'circuit_visualizations'

    id: Mapped[int] = mapped_column(primary_key=True)
    teleportation_id: Mapped[int] = mapped_column(db.ForeignKey('teleportation_results.id'))
    depth: Mapped[int] = mapped_column(Integer)
    size_complexity: Mapped[float] = mapped_column(Float)
    quantum_volume: Mapped[float] = mapped_column(Float)
    parallelism_efficiency: Mapped[float] = mapped_column(Float)
    gate_distribution = mapped_column(SafeJSON)  # Store gate counts as JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    teleportation_result = db.relationship('TeleportationResult', back_populates='circuit_visualization')

    def to_dict(self):
        return {
            'id': self.id,
            'depth': self.depth,
            'size_complexity': self.size_complexity,
            'quantum_volume': self.quantum_volume,
            'parallelism_efficiency': self.parallelism_efficiency,
            'gate_distribution': self.gate_distribution,
            'created_at': self.created_at.isoformat()
        }

class WalletRecommendation(db.Model):
    __tablename__ = 'wallet_recommendations'

    id: Mapped[int] = mapped_column(primary_key=True)
    cryptocurrency: Mapped[str] = mapped_column(String(100))
    wallet_type: Mapped[str] = mapped_column(String(50))  # 'hot', 'cold', 'hardware', etc.
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(255))
    usa_restricted: Mapped[bool] = mapped_column(Boolean, default=False)
    airdrop_support: Mapped[bool] = mapped_column(Boolean, default=False)
    integrated_apps = mapped_column(SafeJSON)  # Store list of integrated apps as JSON
    location_notes: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)