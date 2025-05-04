import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable

from bleak import BleakClient, BleakScanner, BleakError
from bleak.backends.device import BLEDevice

from config import BluetoothConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bluetooth_manager")

class BluetoothManager:
    """Manages Bluetooth connections and data synchronization for multiple devices"""
    
    def __init__(self, config: BluetoothConfig):
        self.config = config
        self.connected_devices: Dict[str, BleakClient] = {}
        self.device_names: Dict[str, str] = {}
        self.data_callbacks: Dict[str, Callable] = {}
        self.connection_status_callback = None
        self.is_running = False
        self.scan_results: List[BLEDevice] = []
        self.last_received_time: Dict[str, datetime] = {}
        
    async def scan_devices(self, timeout: float = 5.0) -> List[BLEDevice]:
        """Scan for available BLE devices"""
        logger.info(f"Scanning for BLE devices (timeout: {timeout}s)...")
        self.scan_results = await BleakScanner.discover(timeout=timeout)
        logger.info(f"Found {len(self.scan_results)} devices")
        return self.scan_results
    
    def get_device_by_name(self, name: str) -> Optional[BLEDevice]:
        """Find a device by name in scan results"""
        for device in self.scan_results:
            if device.name and name.lower() in device.name.lower():
                return device
        return None
    
    async def connect_to_device(self, device: BLEDevice, name: str, 
                              data_callback: Callable) -> bool:
        """Connect to a specific BLE device"""
        try:
            client = BleakClient(device)
            await client.connect(timeout=self.config.connection_timeout)
            
            # Store device information
            self.connected_devices[device.address] = client
            self.device_names[device.address] = name
            self.data_callbacks[device.address] = data_callback
            self.last_received_time[device.address] = datetime.now()
            
            logger.info(f"Connected to {name} device ({device.address})")
            
            # Setup notifications for the data characteristic
            char_uuid = self.config.ecg_characteristic_uuid if "ecg" in name.lower() else self.config.hrv_characteristic_uuid
            await client.start_notify(char_uuid, 
                                    lambda _, data: self._handle_data(device.address, data))
            
            # Update connection status
            if self.connection_status_callback:
                self.connection_status_callback(name, True)
                
            return True
            
        except BleakError as e:
            logger.error(f"Failed to connect to {name} device: {e}")
            if self.connection_status_callback:
                self.connection_status_callback(name, False)
            return False
    
    def _handle_data(self, device_address: str, data: bytearray):
        """Handle incoming data from a device"""
        # Update last received time for synchronization
        current_time = datetime.now()
        self.last_received_time[device_address] = current_time
        
        # Process data through callback
        if device_address in self.data_callbacks:
            device_name = self.device_names.get(device_address, "Unknown")
            self.data_callbacks[device_address](data, current_time, device_name)
    
    async def disconnect_from_device(self, device_address: str) -> bool:
        """Disconnect from a specific device"""
        if device_address in self.connected_devices:
            client = self.connected_devices[device_address]
            try:
                await client.disconnect()
                logger.info(f"Disconnected from {self.device_names.get(device_address, device_address)}")
                
                # Update connection status
                if self.connection_status_callback:
                    self.connection_status_callback(self.device_names.get(device_address, "Unknown"), False)
                    
                # Clean up
                del self.connected_devices[device_address]
                del self.data_callbacks[device_address]
                del self.last_received_time[device_address]
                if device_address in self.device_names:
                    del self.device_names[device_address]
                    
                return True
            except BleakError as e:
                logger.error(f"Error disconnecting from device: {e}")
        return False
    
    async def disconnect_all(self):
        """Disconnect from all connected devices"""
        for addr in list(self.connected_devices.keys()):
            await self.disconnect_from_device(addr)
    
    def set_connection_status_callback(self, callback: Callable):
        """Set callback for connection status updates"""
        self.connection_status_callback = callback
    
    async def monitor_connections(self):
        """Monitor connections and attempt reconnection if needed"""
        self.is_running = True
        while self.is_running:
            for addr, client in list(self.connected_devices.items()):
                if not client.is_connected:
                    logger.warning(f"Device {self.device_names.get(addr, addr)} disconnected. Attempting reconnection...")
                    
                    # Update connection status
                    if self.connection_status_callback:
                        self.connection_status_callback(self.device_names.get(addr, "Unknown"), False)
                    
                    # Rescan for the device
                    await self.scan_devices()
                    device_name = self.device_names.get(addr, "")
                    device = self.get_device_by_name(device_name)
                    
                    if device:
                        await self.connect_to_device(
                            device, 
                            device_name, 
                            self.data_callbacks.get(addr)
                        )
            await asyncio.sleep(self.config.connection_monitor_interval)
    
    def stop_monitoring(self):
        """Stop the connection monitoring task"""
        self.is_running = False
        
    def get_sync_status(self) -> Dict[str, Tuple[datetime, float]]:
        """Get synchronization status of all connected devices"""
        current_time = datetime.now()
        status = {}
        
        for addr, last_time in self.last_received_time.items():
            time_diff = (current_time - last_time).total_seconds()
            status[self.device_names.get(addr, addr)] = (last_time, time_diff)
        
        return status
    
    def check_sync_deviation(self) -> float:
        """Check time synchronization deviation between devices"""
        if len(self.last_received_time) < 2:
            return 0.0
        
        times = list(self.last_received_time.values())
        max_time = max(times)
        min_time = min(times)
        
        return (max_time - min_time).total_seconds()