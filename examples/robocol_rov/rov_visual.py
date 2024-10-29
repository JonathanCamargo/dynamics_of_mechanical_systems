from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import LVector3
from panda3d.core import TextNode, PandaNode, Vec3
from direct.gui.OnscreenText import OnscreenText
#from zero import ZeroClient
import numpy

assets={'rov':'./assets/rov.glb','ring':'./assets/mountains.glb'}

class Player():
    def __init__(self, gameReference):
        self.manipulator = gameReference.render.attachNewNode(PandaNode("player"))
        self.model = gameReference.loader.loadModel(assets['rov'])
        self.model.reparentTo(self.manipulator)               
        self.defaults = {'acceleration': 3.0, 'turn_speed':1.0,'u_increment':0.005}        
        self.u=[0,0,0,0]
        self.velocity=0
                    
    def update(self, dt, keyMap, sceneRoot):
        orientationQuat = self.manipulator.getQuat(sceneRoot)
        forward = orientationQuat.getForward()
        right = orientationQuat.getRight()

        if keyMap["up"]:
            self.velocity += forward*self.defaults['acceleration']*dt
            walking = True
        if keyMap["down"]:
            self.velocity -= forward*self.defaults['acceleration']*dt
            walking = True
        if keyMap["right"]:
            self.velocity += right*self.defaults['acceleration']*dt
            walking = True
        if keyMap["left"]:
            self.velocity -= right*self.defaults['acceleration']*dt
            walking = True
        if keyMap["turnRight"]:
            self.manipulator.setH(self.manipulator, -self.defaults['turn_speed']*dt)
        if keyMap["turnLeft"]:
            self.manipulator.setH(self.manipulator, self.defaults['turn_speed']*dt)

        self.manipulator.setPos(self.manipulator.getPos() + self.velocity*dt)        
   
class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.ring = self.loader.loadModel(assets['ring'])
        self.ring.reparentTo(self.render)
              
        self.disableMouse()        

        # Input: a basic key-map, followed by some key-events
        self.keyMap = {
            "up" : False,
            "down" : False,                      
            "left" : False,            
            "right" : False,            
            "turnRight" : False,         
            "turnLeft" : False,         
        }
        
        self.setBackgroundColor(24/255, 153/255, 221/255)
        self.setupLights()

        self.accept("up", self.updateKeyMap, ["up", True])
        self.accept("down", self.updateKeyMap, ["down", True])
        self.accept("left", self.updateKeyMap, ["left", True])
        self.accept("right", self.updateKeyMap, ["right", True])                        
        self.accept("turnLeft", self.updateKeyMap, ["w", True])
        self.accept("turnRight", self.updateKeyMap, ["s", True])
        self.accept("escape", self.userExit)

        self.title = OnscreenText(text="ROBOCOL - ROV Submarino",
                                  parent=base.a2dBottomRight, style=1,
                                  fg=(0, 0, 0, 1), pos=(-0.2, 0.8),
                                  align=TextNode.ARight, scale=.09)

        self.player = Player(self)
        self.taskMgr.add(self.mytask, "mytask")
        self.updateTask = self.taskMgr.add(self.update, "update")

    def updateKeyMap(self, controlName, controlState):
        self.keyMap[controlName] = controlState

    def update(self, task):
        dt = globalClock.getDt()
        self.player.update(dt, self.keyMap, self.render)
        return task.cont

    def mytask(self,task):        
        #Ejecutar la simulación para el tiempo actual
        # Tomar el estado anterior, calcular 
        return task.cont

    def updateCameraInfo(self, task):
        # Get camera position and orientation
        cam_pos = self.camera.getPos()
        cam_hpr = self.camera.getHpr()                
        return task.cont

    # This function sets up the lighting
    def setupLights(self):
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.8, .8, .75, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setPoint(LVector3(1, 1, 1))
        directionalLight.setDirection(LVector3(0, 0, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))        
        self.render.setLight(self.render.attachNewNode(directionalLight))

app = MyApp()
app.run()