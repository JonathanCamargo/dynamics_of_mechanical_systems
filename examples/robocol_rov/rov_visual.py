import sys
sys.path.append('../../src/')
from SimpleThirdPersonCamera.SimpleThirdPersonCamera import SimpleThirdPersonCamera, SIMPLE_THIRD_PERSON_CAMERA_SIDE_CENTER
from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import LVector3
from panda3d.core import TextNode, PandaNode, Vec3
from direct.gui.OnscreenText import OnscreenText
from zero import ZeroClient
import asyncio


assets={'rov':'./assets/rov.glb','ring':'./assets/mountains.glb'}
zero_client = ZeroClient("localhost", 5559)



class Player():
    def __init__(self, gameReference):
        self.manipulator = gameReference.render.attachNewNode(PandaNode("player"))
        self.model = gameReference.loader.loadModel(assets['rov'])
        self.model.reparentTo(self.manipulator)
        
        self.cameraController= SimpleThirdPersonCamera(-20, 1, 0.2, 0.8, 1.0, 10.0,
                                                                      SIMPLE_THIRD_PERSON_CAMERA_SIDE_CENTER,
                                                                      self.manipulator,
                                                                      gameReference.camera)        
             
        
        self.defaults = {'acceleration': 3.0, 'turn_speed':1.0,'u_increment':0.005}
        self.InitializeFromServer()    
        self.u=[0,0,0,0]
            

    def InitializeFromServer(self):
        q=zero_client.call("getq","dummy")
        pos=self.manipulator.getPos()
        self.manipulator.setPos(q[1],q[2],pos[2])
    
    def update(self, dt, keyMap, sceneRoot):        

        if keyMap["1up"]:
            self.u[0]=self.u[0]+self.defaults['u_increment']
        if keyMap["1down"]:
            self.u[0]=self.u[0]-self.defaults['u_increment']        
        if keyMap["2up"]:
            self.u[1]=self.u[1]+self.defaults['u_increment']
        if keyMap["2down"]:
            self.u[1]=self.u[1]-self.defaults['u_increment']        
        if keyMap["3up"]:
            self.u[2]=self.u[2]+self.defaults['u_increment']
        if keyMap["3down"]:
            self.u[2]=self.u[2]-self.defaults['u_increment']                    
        if keyMap["4up"]:
            self.u[3]=self.u[3]+self.defaults['u_increment']
        if keyMap["4down"]:
            self.u[3]=self.u[3]-self.defaults['u_increment']        
        
        print(self.u)
        msg={'u':self.u,'dt':dt}
        q=zero_client.call("advance",msg)        
        
        pos=self.manipulator.getPos()
        pos[0]=q[1]
        pos[1]=q[2]
        
        self.manipulator.setPos(pos)
        self.manipulator.setH(q[0])
        self.cameraController.update(dt, sceneRoot)


    def update2(self, dt, keyMap, sceneRoot):
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
            self.manipulator.setH(self.manipulator, -self.turnSpeed*dt)
        if keyMap["turnLeft"]:
            self.manipulator.setH(self.manipulator, self.turnSpeed*dt)


        self.manipulator.setPos(self.manipulator.getPos() + self.velocity*dt)
        self.cameraController.update(dt, sceneRoot)

   # A method to clean up when destroying this object
    def cleanup(self):
        if self.cameraController is not None:
            self.cameraController.cleanup()
            self.cameraController = None

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.ring = self.loader.loadModel(assets['ring'])
        self.ring.reparentTo(self.render)
              
        self.disableMouse()

        self.exitFunc = self.cleanup

        # Input: a basic key-map, followed by some key-events
        self.keyMap = {
            "1up" : False,
            "1down" : False,
            "2up" : False,
            "2down" : False,
              "3up" : False,
            "3down" : False,
              "4up" : False,
            "4down" : False,

          
        }

        #Hpr: YAW,PITCH,ROLL (Z,X,Y)
        #self.camera.setPosHpr(1, -15, 6, 2, -22, 0)
        
        self.setBackgroundColor(24/255, 153/255, 221/255)
        self.setupLights()

        self.accept("1", self.updateKeyMap, ["1up", True])
        self.accept("1-up", self.updateKeyMap, ["1up", False])
        self.accept("q", self.updateKeyMap, ["1down", True])
        self.accept("q-up", self.updateKeyMap, ["1down", False])
        self.accept("2", self.updateKeyMap, ["2up", True])
        self.accept("2-up", self.updateKeyMap, ["2up", False])
        self.accept("w", self.updateKeyMap, ["2down", True])
        self.accept("w-up", self.updateKeyMap, ["2down", False])
        self.accept("3", self.updateKeyMap, ["3up", True])
        self.accept("3-up", self.updateKeyMap, ["3up", False])
        self.accept("e", self.updateKeyMap, ["3down", True])
        self.accept("e-up", self.updateKeyMap, ["3down", False])
        self.accept("4", self.updateKeyMap, ["4up", True])
        self.accept("4-up", self.updateKeyMap, ["4up", False])
        self.accept("r", self.updateKeyMap, ["4down", True])
        self.accept("r-up", self.updateKeyMap, ["4down", False])
        

        
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

        # Update text content with camera pose information
        #self.camera_info_text.setText(f"Camera Pose:\nPosition: {cam_pos}\nOrientation: {cam_hpr}")        
        return task.cont

    def cleanup(self):
        if self.pusher is not None:
            self.pusher.removeCollider(self.player.collider)
            self.pusher = None

        if self.cTrav is not None:
            self.cTrav.removeCollider(self.player.collider)
            self.cTrav = None

        if self.player is not None:
            self.player.cleanup()
            self.player = None



    # This function sets up the lighting
    def setupLights(self):
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.8, .8, .75, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setPoint(LVector3(1, 1, 1))
        directionalLight.setDirection(LVector3(0, 0, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        #self.render.setLight(self.render.attachNewNode(ambientLight))
        self.render.setLight(self.render.attachNewNode(directionalLight))


app = MyApp()
app.run()