from direct.showbase.ShowBase import ShowBase
from panda3d.core import TextNode, InputDevice
from direct.gui.OnscreenText import OnscreenText

class ControllerDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # Detect and enable the first connected game controller
        self.accept("connect-device", self.add_controller)
        self.accept("disconnect-device", self.remove_controller)
        
        self.controller = None
        self.controller_text = OnscreenText(text="No Controller Detected",
                                            pos=(-0.5, 0.8),
                                            scale=0.07,
                                            fg=(1, 1, 1, 1),
                                            align=TextNode.ALeft)

        self.taskMgr.add(self.update_controller, "update_controller")
    
    def add_controller(self, device):
        if device.device_class == InputDevice.DeviceClass.gamepad and self.controller is None:
            self.controller = device
            self.attachInputDevice(device)  # Attach the controller to receive input updates
            self.controller_text.setText("Controller Connected")
            print("Controller Connected")
    
    def remove_controller(self, device):
        if device == self.controller:
            self.controller = None
            self.detachInputDevice(device)  # Detach the controller so its events stop updating
            self.controller_text.setText("Controller Disconnected")
            print("Controller Disconnected")
    
    def update_controller(self, task):
        if self.controller:
            state = ""
            for button in self.controller.buttons:
                if button.pressed:
                    state += f"Button {button} pressed\n"
            
            for axis in self.controller.axes:
                value = axis.value
                if abs(value) > 0.1:  # Dead zone filter
                    state += f"Axis {axis}: {value:.2f}\n"
            
            self.controller_text.setText(state if state else "Controller Connected")
        
        return task.cont

if __name__ == "__main__":
    app = ControllerDemo()
    app.run()
