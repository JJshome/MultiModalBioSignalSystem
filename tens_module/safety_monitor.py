import logging
import time
import threading
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safety_monitor")

class SafetyMonitor:
    """Monitors safety parameters for TENS stimulation"""
    
    def __init__(self):
        # Safety limits
        self.max_intensity = 50.0  # mA
        self.max_frequency = 120.0  # Hz
        self.max_pulse_width = 300.0  # µs
        self.max_session_duration = 60.0  # minutes
        self.min_rest_between_sessions = 60.0  # minutes
        
        # Session history
        self.session_history = []
        
        # Current status
        self.safety_status = {
            'status': 'OK',
            'warnings': [],
            'errors': []
        }
        
        # Threading
        self.monitor_thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start the safety monitor"""
        logger.info("Starting safety monitor")
        
        # Clear status
        self.safety_status = {
            'status': 'OK',
            'warnings': [],
            'errors': []
        }
        
        # Start monitoring thread
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Safety monitor started successfully")
    
    def stop(self):
        """Stop the safety monitor"""
        logger.info("Stopping safety monitor")
        
        # Signal monitoring thread to stop
        self.stop_event.set()
        
        # Wait for thread to end
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Safety monitor stopped successfully")
    
    def check_stimulation_safety(self, parameters: Dict) -> Dict:
        """Check if stimulation parameters are safe"""
        # Extract parameters
        frequency = parameters.get('frequency', 0.0)
        pulse_width = parameters.get('pulse_width', 0.0)
        intensity = parameters.get('intensity', 0.0)
        duration = parameters.get('duration', 0.0)
        
        # Reset issues list
        issues = []
        
        # Check individual parameter limits
        if intensity > self.max_intensity:
            issues.append(f"Intensity ({intensity} mA) exceeds maximum limit ({self.max_intensity} mA)")
        
        if frequency > self.max_frequency:
            issues.append(f"Frequency ({frequency} Hz) exceeds maximum limit ({self.max_frequency} Hz)")
        
        if pulse_width > self.max_pulse_width:
            issues.append(f"Pulse width ({pulse_width} µs) exceeds maximum limit ({self.max_pulse_width} µs)")
        
        if duration > self.max_session_duration:
            issues.append(f"Session duration ({duration} min) exceeds maximum limit ({self.max_session_duration} min)")
        
        # Check combination safety
        # High intensity and pulse width combination can be dangerous
        if intensity > self.max_intensity * 0.8 and pulse_width > self.max_pulse_width * 0.8:
            issues.append("Dangerous combination of high intensity and high pulse width")
        
        # Check session timing limitations based on history
        current_time = time.time()
        for session in self.session_history:
            end_time = session.get('end_time', 0)
            if current_time - end_time < self.min_rest_between_sessions * 60:  # Convert to seconds
                time_since_last = (current_time - end_time) / 60.0  # in minutes
                issues.append(f"Insufficient rest time since last session ({time_since_last:.1f} min vs. required {self.min_rest_between_sessions} min)")
                break
        
        # Determine overall safety status
        is_safe = len(issues) == 0
        
        # Update internal safety status
        if not is_safe:
            self.safety_status['status'] = 'WARNING'
            self.safety_status['warnings'] = issues
        
        return {
            'safe': is_safe,
            'message': '; '.join(issues) if issues else "All parameters within safe limits",
            'issues': issues
        }
    
    def record_session(self, start_time: float, end_time: float, parameters: Dict):
        """Record a completed stimulation session"""
        session_record = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': (end_time - start_time) / 60.0,  # in minutes
            'parameters': parameters.copy()
        }
        
        # Add to session history
        self.session_history.append(session_record)
        
        # Keep only the last 10 sessions
        if len(self.session_history) > 10:
            self.session_history = self.session_history[-10:]
        
        logger.info(f"Recorded session: {session_record['duration']:.1f} minutes")
    
    def check_electrode_safety(self, contact_status: Dict) -> Dict:
        """Check electrode contact safety"""
        # Extract contact information
        contact_ok = contact_status.get('contact_ok', False)
        impedance = contact_status.get('impedance', float('inf'))
        
        if not contact_ok:
            return {
                'safe': False,
                'message': "Poor electrode contact detected",
                'issues': ["Electrode contact issue", f"Impedance: {impedance} ohms"]
            }
        
        return {
            'safe': True,
            'message': "Electrode contact OK",
            'issues': []
        }
    
    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        return self.safety_status.copy()
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("Safety monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Periodic safety checks could be done here
                # This is mostly a placeholder for more sophisticated monitoring
                
                # Check for session expiry
                current_time = time.time()
                for session in self.session_history:
                    # Check if sessions are getting too frequent
                    if len(self.session_history) > 3:
                        recent_sessions = self.session_history[-3:]
                        total_duration = sum([s.get('duration', 0) for s in recent_sessions])
                        if total_duration > self.max_session_duration * 1.5:
                            warning = "Excessive stimulation detected in recent sessions"
                            if warning not in self.safety_status['warnings']:
                                self.safety_status['warnings'].append(warning)
                                self.safety_status['status'] = 'WARNING'
                                logger.warning(warning)
                
                # Sleep for a while before next check
                time.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(60.0)  # Longer sleep on error
        
        logger.info("Safety monitoring loop ended")